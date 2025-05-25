FROM python:3.9.7

WORKDIR /szu-app

RUN apt-get update && apt-get install -y vim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
