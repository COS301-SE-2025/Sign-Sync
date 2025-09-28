FROM sign-sync-base

WORKDIR /app

COPY . .

EXPOSE 8007

CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8007"]

