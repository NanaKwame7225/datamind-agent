FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# index.html must be present in the root of the repo
# Railway will fail the healthcheck if it's missing — double check it's committed

EXPOSE 8000

# $PORT is injected by Railway at runtime
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
