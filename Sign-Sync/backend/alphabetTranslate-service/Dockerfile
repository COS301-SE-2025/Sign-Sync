FROM sign-sync-base

WORKDIR /app

COPY . .

EXPOSE 8000

CMD ["uvicorn", "translator_api:app", "--host", "0.0.0.0", "--port", "8000"]

