FROM sign-sync-base

WORKDIR /app

COPY . .

# Read port from env or default to 8000
ENV PORT=${PORT:-8000}
EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn translator_api:app --host 0.0.0.0 --port $PORT"]
