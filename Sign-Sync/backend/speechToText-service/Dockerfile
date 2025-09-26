FROM sign-sync-base

WORKDIR /app

# Install ffmpeg and clean up to reduce image size
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8003

CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8003"]

