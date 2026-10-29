# Reproducible environment for expkit. Data is downloaded at run time (not baked into the image).
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends make libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install uv==0.13.5
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]" gsdesign==0.1.5
COPY . .
# docker build -t expkit . && docker run --rm -v $PWD/data:/app/data -v $PWD/results:/app/results expkit make all
CMD ["make", "all"]
