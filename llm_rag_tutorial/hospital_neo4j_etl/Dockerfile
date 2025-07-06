## Start with docker image
# FROM python:3.12.3-alpine3.20
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

## Working directory is app in Docker container
WORKDIR /app

## Copy src folder contents into /app
COPY ./src/ /app

## Copy pyproject.toml into /code folder
# COPY ./pyproject.toml /code/pyproject.toml

# requirements.txt created from pyproject.toml using uv
WORKDIR /code
# COPY ./requirements.txt /code/requirements.txt
COPY ./requirements.txt ./requirements.txt

## Install dependencies from pyproject.toml
# RUN pip install /code/.
# RUN pip install -r ./requirements.txt
# RUN uv pip install --no-installer-metadata --no-compile-bytecode --python-platform linux --python 3.12 --system -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
CMD ["sh", "entrypoint.sh"]