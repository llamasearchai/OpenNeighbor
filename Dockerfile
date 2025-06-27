FROM python:3.10-slim

# Set metadata
LABEL maintainer="Nik Jois <nikjois@llamasearch.ai>"
LABEL description="OpenNeighbor: Production-Grade Neighborhood-Aware Recommendation System"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd -r openneighbor && useradd -r -g openneighbor openneighbor

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/outputs
RUN chown -R openneighbor:openneighbor /app

# Switch to non-root user
USER openneighbor

# Expose port for API server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import openneighbor; print('OpenNeighbor is healthy')" || exit 1

# Default command
CMD ["python", "openneighbor_cli.py", "serve", "--host", "0.0.0.0", "--port", "8000"] 