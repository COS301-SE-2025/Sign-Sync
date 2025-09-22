FROM sign-sync-base

WORKDIR /app

#Download the spaCy English model
RUN python -m spacy download en_core_web_sm && \
    rm -rf /root/.cache

COPY . .

#EXPOSE 8002

#CMD ["uvicorn", "textConversion_api:app", "--host", "0.0.0.0", "--port", "8002"]

# Read port from env or default to 8002
ENV PORT=${PORT:-8002}
EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn textConversion_api:app --host 0.0.0.0 --port $PORT"]
