FROM sign-sync-base

WORKDIR /app

COPY . .

EXPOSE 8008

CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8008"]


