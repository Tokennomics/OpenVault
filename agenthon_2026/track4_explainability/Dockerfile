# Reproducible scoring container for the minimal Track 4 baseline.
# The baseline is pure standard library, so the image is tiny and runs offline.
FROM python:3.13-slim

# Required on every submission image by the published contract (SUBMISSION_CLI.md); gate
# g0_integrity checks for it. The reference baseline satisfies the same rule it asks
# participants to follow.
LABEL qfbench2.interface_version="2.0"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY baseline_agent ./baseline_agent
COPY baseline_agent.py .

# The harness runs the image as:
#
#   docker run --rm --network qfb2-eval \
#       -v <unit>:/input:ro -v <run>/output:/output \
#       <IMAGE> analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json
#
# The verb `analyze` is the container COMMAND, so with this ENTRYPOINT it arrives as
# baseline_agent.py's first positional argument — which is why baseline_agent/cli.py declares a
# leading `verb` positional. An image that does not consume it exits 2 before reading any input.
# The alternative shape (no ENTRYPOINT, `analyze` as an executable on PATH) is equally valid and
# is what the Track 3 reference baseline does.
ENTRYPOINT ["python", "baseline_agent.py"]
CMD ["analyze", "--help"]
