# Chỉ định platform để tránh lỗi exec format
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Cài đặt các dependencies hệ thống (gộp lại để giảm layers)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Cài đặt Python dependencies với timeout và retry
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy toàn bộ source code
COPY . .

# Expose port cho Streamlit (mặc định là 8501)
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')" || exit 1

# Debug: List files để kiểm tra
RUN ls -la && ls -la pages/ || echo "pages folder not found"

# Command để chạy ứng dụng
CMD ["streamlit", "run", "GiaoDien/Web.py", "--server.port=8501", "--server.address=0.0.0.0"]
