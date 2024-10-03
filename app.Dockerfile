FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.3
RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION}
ENV PATH="${PATH}:/root/.local/bin"

# Copy project files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install

# Copy source code
COPY . .

ENV PROD=true
ENV PYTHONPATH=/

# Set environment variables for Streamlit production
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8000
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_LOG_LEVEL=error

CMD ["poetry", "run", "streamlit", "run", "src/deployments/app/__main__.py", "--server.port", "8000", "--server.address", "0.0.0.0"]


EXPOSE 8000