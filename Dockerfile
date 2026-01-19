FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --no-cache-dir -r requirements.txt


COPY . .

# Expose port (defaults to 8080)
EXPOSE 8080

# Production Entrypoint
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} app:app
