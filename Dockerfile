FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_STORAGE_DIR=/app/runtime

WORKDIR /app

COPY requirements.txt requirements-extended.txt ./
ARG INSTALL_EXTENDED=false
RUN if [ "$INSTALL_EXTENDED" = "true" ]; then \
      pip install -r requirements-extended.txt; \
    else \
      pip install -r requirements.txt; \
    fi

COPY . .
RUN mkdir -p /app/runtime

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
