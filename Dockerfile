
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.text ./requirements/base.text

RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --timeout 180 --retries 10 -r requirements/base.text

COPY app/ ./app/
COPY src/ ./src/
COPY models/feature_list.pkl ./models/feature_list.pkl
COPY models/imputer.pkl ./models/imputer.pkl
COPY config/ ./config/
COPY gx/ ./gx/

RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
