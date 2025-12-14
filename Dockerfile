FROM python:3.13-slim

ARG RYE_VERSION="0.44.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RYE_HOME=/opt/rye \
    PATH="/opt/rye/shims:/opt/rye/bin:${PATH}" \
    PYTHONPATH=/app/src \
    DJANGO_SETTINGS_MODULE=project.settings

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSf https://rye-up.com/install.sh | RYE_HOME=${RYE_HOME} RYE_VERSION=${RYE_VERSION} bash -s -- -y

COPY pyproject.toml requirements.lock requirements-dev.lock /app/

RUN rye sync --no-dev --toolchain=cpython@3.13

COPY . /app

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

CMD ["python", "src/webapp/manage.py", "runserver", "0.0.0.0:8000"]

