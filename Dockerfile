FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including LaTeX
RUN apt-get update && apt-get install -y \
    build-essential \
    texlive-full \
    && rm -rf /var/lib/apt/lists/* \
    && which pdflatex && pdflatex --version

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create necessary directories
RUN mkdir -p /app/temp /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Run the FastAPI application
CMD ["python", "-m", "uvicorn", "cv_service:app", "--host", "0.0.0.0", "--port", "8000"]
