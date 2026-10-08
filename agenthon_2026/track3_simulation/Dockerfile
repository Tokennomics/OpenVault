# Track 3 baseline image: pinned upstream ABIDES + the `abides_fork` simulate adapter.
#
# This is the pre-built baseline the organizers ship. ABIDES itself is NOT vendored
# in this repo; it is fetched at build time at the pinned commit (see README.md) and
# the `abides_fork` adapter (this directory) layers the canonical `simulate` verb on
# top of the unmodified engine.
#
# Build context is this directory (baselines/). Build for the evaluation platform:
#
#   docker build --platform=linux/amd64 -t track3-abides-baseline:latest baselines/
#
# The evaluation harness (regression_suite/run_regression.py) runs it, network
# disabled, exactly as a participant submission would be run:
#
#   docker run --rm --network none \
#       -v <in>:/input:ro -v <out>:/output \
#       track3-abides-baseline:latest \
#       simulate --config /input/scenario.json --out /output/trace.parquet
# NOTE — deliberate exception to the org's Python 3.13 standard. Do NOT bump this.
# ABIDES requires pandas 1.x, and the pinned stack below (numpy==1.26.4, pandas==1.5.3) publishes
# no cp313 wheels; this is also the exact stack that generated the frozen reference traces. The
# image is self-contained (the harness only reads the parquet it writes), so it does not constrain
# the 3.13 evaluation container or participant submission images.
FROM --platform=linux/amd64 python:3.11-slim

# Required on every submission image by the published contract (SUBMISSION_CLI.md and the
# CodaBench Submission Format page). The reference baseline satisfies the same rule it asks
# participants to follow.
LABEL qfbench2.interface_version="2.0"

# ABIDES is pinned to the canonical Track 3 baseline commit (see baselines/README.md).
# Override at build time only to test a different upstream pin.
ARG ABIDES_REPO=https://github.com/jpmorganchase/abides-jpmc-public.git
ARG ABIDES_COMMIT=f9cbe51342b7dedd9587e4e069040d68a5c6477f

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt

# Pin the exact stack that generated the reference traces. ABIDES needs pandas 1.x;
# the eval container's pandas 2.x is irrelevant because this image is self-contained
# and the harness only *reads* the parquet it writes. coloredlogs is a runtime import
# of abides_core.abides (it is NOT declared in ABIDES's setup.cfg, so the --no-deps
# install below will not pull it — it must be installed explicitly here).
RUN pip install \
        numpy==1.26.4 \
        pandas==1.5.3 \
        scipy==1.17.1 \
        pyarrow==15.0.2 \
        coloredlogs==15.0.1

# Fetch ABIDES at the pinned commit and apply our build-time overlays from patches/ to a
# FRESH upstream clone (upstream repo untouched; nothing modified is committed anywhere):
#   1. order_size_model.pomegranate-free.patch — drop the unbuildable pomegranate dependency
#      (drop-in NumPy mixture sampler).
#   2. kernel_message_ledger.patch — observation-only instrumentation that surfaces per-message
#      send/recv/latency/causal_parent/seq metadata in end_state for the v2 enriched trace
#      schema. Zero extra RNG; no simulation-logic, ordering, or RNG change.
#   3. exchange_protocol_stp.patch — adds an OPT-IN self-trade-prevention policy to the exchange
#      (ExchangeAgent.stp_policy, default None = legacy behavior; OrderBook honors cancel_newest /
#      cancel_oldest). Inert unless a scenario opts in, so all existing references are unchanged.
#   4. oracle_scheduled_jump.patch — adds an OPT-IN deterministic fundamental jump (reactive-agent
#      intervention) to SparseMeanRevertingOracle. RNG-neutral; absent -> legacy, refs unchanged.
# git is installed and removed inside one layer so it does not bloat the image.
COPY patches/ /tmp/patches/
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && git clone "${ABIDES_REPO}" /tmp/abides \
    && git -C /tmp/abides checkout "${ABIDES_COMMIT}" \
    && git -C /tmp/abides apply /tmp/patches/order_size_model.pomegranate-free.patch \
    && git -C /tmp/abides apply /tmp/patches/kernel_message_ledger.patch \
    && git -C /tmp/abides apply /tmp/patches/exchange_protocol_stp.patch \
    && git -C /tmp/abides apply /tmp/patches/oracle_scheduled_jump.patch \
    && pip install --no-deps /tmp/abides/abides-core /tmp/abides/abides-markets \
    && apt-get purge -y git \
    && apt-get autoremove -y \
    && rm -rf /tmp/abides /tmp/patches /var/lib/apt/lists/*

# The Track 3 `simulate` adapter (component-1 classical baseline) and its CLI shims. `simulate` is the
# single-scenario verb; `simulate-batch` is the batched multi-scenario verb (BatchMarketSim / family GB):
# it runs N sub-scenarios in one process and writes one output subdir each + batch_events.json.
COPY abides_fork /opt/abides_fork
COPY simulate /usr/local/bin/simulate
COPY simulate-batch /usr/local/bin/simulate-batch
RUN chmod +x /usr/local/bin/simulate /usr/local/bin/simulate-batch

WORKDIR /work
# No ENTRYPOINT: the harness passes `simulate --config ... --out ...` as the command.
CMD ["simulate", "--help"]
