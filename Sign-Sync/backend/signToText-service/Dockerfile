FROM sign-sync-base

WORKDIR /app

COPY . .

EXPOSE 8006

CMD ["uvicorn", "signToText_API:app", "--host", "0.0.0.0", "--port", "8006"]