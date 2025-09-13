# Multi-stage Dockerfile for YOLOS project
# Supports both development and production environments

# Base image with Python and system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r yolos && useradd -r -g yolos yolos

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit \
    jupyter \
    ipython

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e ".[dev]"

# Change ownership to non-root user
RUN chown -R yolos:yolos /app
USER yolos

# Expose ports
EXPOSE 8000 8888

# Default command for development
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8000"]

# Production stage
FROM base as production

# Install only production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY src/ ./src/
COPY config/ ./config/
COPY web/ ./web/
COPY setup.py pyproject.toml MANIFEST.in ./

# Install package
RUN pip install --no-cache-dir .

# Create directories for runtime data
RUN mkdir -p /app/logs /app/models /app/data /app/temp

# Change ownership to non-root user
RUN chown -R yolos:yolos /app
USER yolos

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["python", "-m", "web.app"]

# GPU-enabled stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create non-root user
RUN groupadd -r yolos && useradd -r -g yolos yolos

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY web/ ./web/
COPY setup.py pyproject.toml MANIFEST.in ./

# Install package with GPU support
RUN pip install --no-cache-dir ".[gpu]"

# Create directories
RUN mkdir -p /app/logs /app/models /app/data /app/temp

# Change ownership
RUN chown -R yolos:yolos /app
USER yolos

# Expose port
EXPOSE 8000

# GPU command
CMD ["python", "-m", "web.app"]

# Raspberry Pi stage
FROM python:3.10-slim as raspberry

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install Raspberry Pi specific dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r yolos && useradd -r -g yolos yolos

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY setup.py pyproject.toml MANIFEST.in ./

# Install with Raspberry Pi support
RUN pip install --no-cache-dir ".[raspberry]"

# Create directories
RUN mkdir -p /app/logs /app/models /app/data /app/temp

# Change ownership
RUN chown -R yolos:yolos /app
USER yolos

# Expose port
EXPOSE 8000

# Raspberry Pi command
CMD ["python", "-m", "recognition.cli"]