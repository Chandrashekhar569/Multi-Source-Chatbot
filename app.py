import os
import uuid
import json
import logging
import threading
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiohttp
from bs4 import BeautifulSoup
from io import BytesIO
from pypdf import PdfReader

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

# --- Production-Ready Improvements ---
# 1. API Key for Security
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 2. Thread-safe lock for in-memory vectorstore
vectorstore_lock = threading.Lock()

# 3. Input validation constants
MAX_FILE_SIZE_MB = 25
MAX_URL_CONTENT_LENGTH = 10 * 1024 * 1024 # 10 MB

# --- Responsible AI & Chatbot Flow Policy ---
RESPONSIBLE_AI_TEMPLATE = """You are a helpful and responsible AI chatbot. Your goal is to answer questions in a friendly, conversational manner based ONLY on the provided context.

**Policy:**
1. **Strictly Contextual:** Base your entire answer on the text provided in the "Context" section below. Make it easy to read and user-friendly, like a chatbot.
2. **Acknowledge Limits:** If the answer is not found in the context, politely say: "I'm sorry, I could not find an answer in the provided documents." Do not guess or fabricate information.
3. **Be Harmless and Unbiased:** Always respond politely, neutrally, and avoid harmful, unethical, or biased content.
4. **Engage Naturally:** Respond like a helpful assistant, using short, clear sentences. Provide guidance, examples, or references if available in the context.

---
**Context:**
{context}
---

**User Question:**
{question}

**Chatbot Response:**
"""

RESPONSIBLE_AI_PROMPT = PromptTemplate(
    template=RESPONSIBLE_AI_TEMPLATE, input_variables=["context", "question"]
)

# --- Original Configuration ---
DATA_DIR = os.getenv('DATA_DIR', 'data_store')
INDEX_DIR = os.path.join(DATA_DIR, 'faiss_index')
LOG_FILE = os.getenv('LOG_FILE', 'logs/queries.log')
RAG_TOP_K = int(os.getenv('RAG_TOP_K', '5'))
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Logging setup
logger = logging.getLogger('ms_chatbot')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(formatter)
logger.addHandler(fh)

# Initialize LangChain components
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY or ''
embeddings = OpenAIEmbeddings()

# Load or create FAISS vectorstore
if os.path.exists(INDEX_DIR):
    with vectorstore_lock:
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = None

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

app = FastAPI(title="Multi-Source Chatbot API (LangChain)")

async def get_api_key(api_key: str = Security(api_key_header)):
    if API_KEY and api_key == API_KEY:
        return api_key
    if not API_KEY:
        return None
    raise HTTPException(status_code=403, detail="Could not validate credentials")

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = RAG_TOP_K

@app.get('/health')
async def health():
    with vectorstore_lock:
        size = len(vectorstore.index_to_docstore_id) if vectorstore else 0
    return {'status': 'ok', 'index_size': size}

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

async def fetch_url_text(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=20) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=400, detail=f'Failed to fetch {url}, status {resp.status}')
            if resp.content_length and resp.content_length > MAX_URL_CONTENT_LENGTH:
                raise HTTPException(status_code=413, detail=f"URL content is too large (>{MAX_URL_CONTENT_LENGTH/1024/1024} MB).")
            html = await resp.text()
    soup = BeautifulSoup(html, 'lxml')
    for s in soup(['script', 'style', 'noscript']):
        s.decompose()
    body = soup.body
    text = body.get_text(separator=' ', strip=True) if body else soup.get_text(separator=' ', strip=True)
    return text

def add_chunks(chunks: List[str], source_type: str, source_name: str, source_id: str):
    global vectorstore
    docs = [Document(page_content=c, metadata={"source_type": source_type, "source_name": source_name, "source_id": source_id}) for c in chunks]
    with vectorstore_lock:
        if vectorstore is None:
            vectorstore = FAISS.from_documents(docs, embeddings)
        else:
            vectorstore.add_documents(docs)
        
        if vectorstore:
            vectorstore.save_local(INDEX_DIR)
            logger.info(f"Saved vectorstore to {INDEX_DIR}")

@app.post('/ingest/document', dependencies=[Depends(get_api_key)])
async def ingest_document(file: UploadFile = File(...), source_name: Optional[str] = Form(None)):
    filename = file.filename
    source_name = source_name or filename
    
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File size exceeds limit of {MAX_FILE_SIZE_MB} MB.")

    text = ''
    if filename.lower().endswith('.pdf'):
        try:
            reader = PdfReader(BytesIO(content))
            pages = [p.extract_text() or '' for p in reader.pages]
            text = ''.join(pages)
        except Exception as e:
            logger.exception('PDF parse error')
            raise HTTPException(status_code=400, detail=f'PDF parse error: {e}')
    elif filename.lower().endswith(('.txt', '.md')):
        text = content.decode('utf-8', errors='ignore')
    else:
        raise HTTPException(status_code=400, detail='Unsupported file type. Accepts PDF, TXT, MD')

    chunks = chunk_text(text)
    add_chunks(chunks, source_type='document', source_name=source_name, source_id=filename)
    logger.info(f'Ingested document: {source_name}, chunks={len(chunks)}')
    return {'status': 'success', 'ingested_chunks': len(chunks), 'source_name': source_name}

@app.post('/ingest/url', dependencies=[Depends(get_api_key)])
async def ingest_url(url: str = Form(...), source_name: Optional[str] = Form(None)):
    source_name = source_name or url
    text = await fetch_url_text(url)
    chunks = chunk_text(text)
    add_chunks(chunks, source_type='web', source_name=source_name, source_id=url)
    logger.info(f'Ingested URL: {source_name}, chunks={len(chunks)}')
    return {'status': 'success', 'ingested_chunks': len(chunks), 'source_name': source_name}

class JSONRecord(BaseModel):
    records: List[Dict[str, Any]]
    source_name: Optional[str] = None

@app.post('/ingest/json', dependencies=[Depends(get_api_key)])
async def ingest_json(payload: JSONRecord):
    source_name = payload.source_name or 'json_records'
    count = 0
    for rec in payload.records:
        txt = json.dumps(rec, ensure_ascii=False)
        chunks = chunk_text(txt)
        add_chunks(chunks, source_type='json', source_name=source_name, source_id=rec.get('id', str(uuid.uuid4())))
        count += len(chunks)
    logger.info(f'Ingested JSON source: {source_name}, chunks={count}')
    return {'status': 'success', 'ingested_chunks': count, 'source_name': source_name}

@app.post('/query', dependencies=[Depends(get_api_key)])
async def query(q: QueryRequest):
    global vectorstore
    if vectorstore is None:
        raise HTTPException(status_code=400, detail='No data ingested yet.')
    
    with vectorstore_lock:
        retriever = vectorstore.as_retriever(search_kwargs={'k': q.top_k})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True,
        chain_type_kwargs={"prompt": RESPONSIBLE_AI_PROMPT}
    )
    result = qa_chain.invoke({'query': q.question})
    answer = result['result']
    sources = [{'source_type': d.metadata.get('source_type'), 'source_name': d.metadata.get('source_name'), 'source_id': d.metadata.get('source_id')} for d in result['source_documents']]

    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps({'question': q.question, 'answer': answer, 'sources': sources}) + '\n')
    except Exception:
        logger.exception('Failed to write query log')

    logger.info(f'Query handled: question="{q.question[:80]}" sources={len(sources)}')
    return {'answer': answer, 'sources': sources}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 8000)))