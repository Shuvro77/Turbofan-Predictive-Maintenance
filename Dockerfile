# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Create Venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install everything together from requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# SAFE OPTIMIZATION: Only remove pycache and compiled python files
RUN find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "__pycache__" -type d -exec rm -rf {} +

# --- Stage 2: Final ---
FROM python:3.12-slim
WORKDIR /app

# Copy the venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files
COPY ./app ./app
COPY ./artifacts ./artifacts

# Clean up pycache to keep it slim
RUN find /opt/venv -type d -name "__pycache__" -exec rm -rf {} +

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Hugging Face uses port 7860
EXPOSE 7860

# Using uvicorn directly (it's in the venv path)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "4"]