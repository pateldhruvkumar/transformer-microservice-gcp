FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
# torch CPU wheel is ~200 MB; caching it avoids re-downloading on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY . .

# Cloud Run injects PORT at runtime; default to 8080
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
