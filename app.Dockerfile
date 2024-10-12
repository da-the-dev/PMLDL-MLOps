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
RUN poetry install --only app

# Copy source code
COPY . .

ENV PROD=true

ENV STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=false

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["poetry", "run", "streamlit", "run", "src/deployments/app/__main__.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

