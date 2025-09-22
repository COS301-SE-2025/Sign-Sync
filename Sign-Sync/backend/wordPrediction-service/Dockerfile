FROM sign-sync-base

WORKDIR /app

COPY . .

#EXPOSE 8005

#CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8005"]

# Read port from env or default to 8005
ENV PORT=${PORT:-8005}
EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn API:app --host 0.0.0.0 --port $PORT"]
