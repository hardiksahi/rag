## Start with docker image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

## Working directory is app in Docker container
WORKDIR /app

## Copy src folder contents into /app
COPY ./src/ /app

WORKDIR /code
COPY ./requirements.txt ./requirements.txt

#RUN uv pip install --no-installer-metadata --no-compile-bytecode --python-platform x86_64-manylinux_2_28 --python 3.12 --system -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
EXPOSE 8000
CMD ["sh", "entrypoint.sh"]