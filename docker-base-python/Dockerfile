#use official Python 3.10 base
FROM python:3.10-slim

#set working directory
WORKDIR /app

#copy requirements file
COPY requirements.txt .

#install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#set environment variable to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

#set environment variable to buffer stdout/stderr
ENV PYTHONUNBUFFERED=1
