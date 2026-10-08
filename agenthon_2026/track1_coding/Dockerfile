FROM python:3.13-slim

LABEL qfbench2.interface_version="2.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy pandas scipy pyarrow scikit-learn pytest jsonschema requests httpx openai

RUN mkdir -p /app/data /app/output /app/tests /logs/verifier /agent

COPY agent.py /agent/agent.py
RUN chmod +x /agent/agent.py

# Verb on PATH
RUN printf '#!/bin/sh\nexec python /agent/agent.py solve "$@"\n' > /usr/local/bin/solve \
 && chmod +x /usr/local/bin/solve

WORKDIR /app

ENTRYPOINT ["python", "/agent/agent.py"]
CMD ["solve", "--task-dir", "/input", "--out", "/app/output"]
