FROM python:3.12-slim

WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/root/.cache/fastembed

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir ".[dev]"

COPY eval ./eval
COPY tests ./tests

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
