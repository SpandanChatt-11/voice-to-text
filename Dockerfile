# Hugging Face Spaces requires the container to listen on port 7860
FROM python:3.11-slim

# System dependencies (ffmpeg for audio decoding, libsndfile for torchaudio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and model checkpoint
COPY app/ ./app/
COPY best_model.pt .

# Hugging Face Spaces expects the app on port 7860
EXPOSE 7860
ENV CHECKPOINT_PATH=/code/best_model.pt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
