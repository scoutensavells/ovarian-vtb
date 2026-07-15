FROM python:3.11.8-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY code ./code
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .
ENTRYPOINT ["msdp-evaluate"]
