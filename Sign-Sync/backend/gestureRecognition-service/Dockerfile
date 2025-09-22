FROM sign-sync-base

WORKDIR /app

COPY . .

#EXPOSE 8008

#CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8008"]

# Read port from env or default to 8008
ENV PORT=${PORT:-8008}
EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn API:app --host 0.0.0.0 --port $PORT"]

