# syntax=docker/dockerfile:1
FROM python:3.12.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Dependencies first -> cached layer as long as requirements.txt is unchanged
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Project package
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# 3) App, Streamlit config, sample data and (optional) pre-trained model
COPY app ./app
COPY .streamlit ./.streamlit
COPY data/sample ./data/sample
COPY models ./models

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
