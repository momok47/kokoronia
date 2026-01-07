FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DJANGO_SETTINGS_MODULE=project.settings

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    portaudio19-dev \
    pkg-config \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.lock requirements-dev.lock README.md /app/

# パッケージ本体を先に配置しておくことで `-e file:.` を解決
COPY src /app/src

RUN pip install --no-cache-dir -r requirements.lock

COPY . /app

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

CMD ["python", "src/webapp/manage.py", "runserver", "0.0.0.0:8000"]

