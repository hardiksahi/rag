FROM python:3.12.3-alpine3.20

WORKDIR /app

COPY ./src/ /app

WORKDIR /code
COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
EXPOSE 8501
CMD ["sh", "entrypoint.sh"]