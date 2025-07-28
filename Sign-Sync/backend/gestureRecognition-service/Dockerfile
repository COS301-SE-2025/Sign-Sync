FROM sign-sync-base

WORKDIR /app

COPY . .

EXPOSE 8003

CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8003"]