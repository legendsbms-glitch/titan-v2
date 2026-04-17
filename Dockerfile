FROM python:3.11-slim

LABEL maintainer="tsk"
LABEL description="TITAN v2.0 — 9-Engine Gold Intelligence System"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create required directories
RUN mkdir -p data db logs journal alerts

# Default: run analysis
EXPOSE 8000 8501

ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

ENTRYPOINT ["python", "main.py"]
CMD ["analyze"]
