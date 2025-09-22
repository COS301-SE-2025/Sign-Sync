FROM sign-sync-base

WORKDIR /app

COPY . .

#EXPOSE 8006

#CMD ["uvicorn", "signToText_API:app", "--host", "0.0.0.0", "--port", "8006"]

# Read port from env or default to 8000
ENV PORT=${PORT:-8006}
EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn signToText_API:app --host 0.0.0.0 --port $PORT"]

