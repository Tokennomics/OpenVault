FROM python:3.13-slim-bookworm

LABEL qfbench2.interface_version="2.0"
LABEL qfbench2.track="analysis"
LABEL qfbench2.verb="analyze"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app
COPY strong_rag_baseline /app/strong_rag_baseline

# The runner calls: analyze --task /input/task.json --corpus /input/corpus --out /output/answer.json
ENTRYPOINT ["python", "-m", "strong_rag_baseline.cli"]
CMD ["analyze", "--help"]
