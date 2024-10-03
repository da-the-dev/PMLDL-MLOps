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
RUN poetry install --only main

# Copy source code
COPY . .

ENV PROD=true
# Run the application with proxy
CMD ["poetry", "run", "python", "-m", "src.deployments.api"]

EXPOSE 8000