# --- Build Stage ---
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies for packages like faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a wheelhouse for the python packages
COPY requirements.txt ./
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# --- Final Stage ---
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels and install them
COPY --from=builder /app/wheels /wheels
COPY requirements.txt ./
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Copy the application code
COPY . .

# Create a non-root user and change ownership of the app directory
RUN useradd -m -d /home/appuser -s /bin/bash appuser && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]