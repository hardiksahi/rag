#!/bin/bash

echo "Running FastAPI application for hospital system rag..."

uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug