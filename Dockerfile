# =========================
# Jarvis Offline Assistant
# =========================
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    wget \
    libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

# Audio dependencies
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
  && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone and build whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp /app/whisper.cpp && \
  cd /app/whisper.cpp && \
  make && \
  ./models/download-ggml-model.sh base.en

# Copy project code
COPY . .

# Default run command
CMD ["python", "main.py"]
