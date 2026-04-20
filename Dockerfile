FROM python:3.10-slim

WORKDIR /app

# Avoid Python buffering issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (important for Pillow & TF)
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy all files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Hugging Face requirement)
EXPOSE 7860

# Run app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
