#!/usr/bin/env bash
# Prove the GPU starter is gate-compatible: build the verification image and check that it still
# reproduces a real public unit BIT-IDENTICALLY (Tier-A: fill sequence + message ledger).
#
# No GPU required. This exercises the CuPy-absent fallback path, which is the point: adding the
# accelerated helper module and the self-checking shim to a submission image must not perturb the
# simulation output. Certifying the CUDA kernel path itself needs the eval box GPU.
#
#   ./verify_gate.sh [unit-dir] [base-image]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
UNIT="${1:-$REPO/units/t3-eq-deterministic-baseline}"
# The tag the documented participant build actually produces:
#     docker build --platform=linux/amd64 -t track3-abides-baseline:latest baselines/
# (baselines/Dockerfile's own header, and the default in baselines/build_and_validate.sh).
#
# This defaulted to `track3-abides-ledger:test`, which NO documented build in this repository
# produces — it is an organizer-side tag used by scripts/build_batch_units.py when regenerating
# references. Running this script after following the published build therefore failed at
# `docker build` with an unresolvable base image. The two names refer to the same artifact: one
# baselines/Dockerfile, which applies kernel_message_ledger.patch and so already emits
# message_trace.parquet.
BASE_IMAGE="${2:-track3-abides-baseline:latest}"
TAG="track3-gpu-starter-verify:test"
# Build and run for the host's own architecture by default. Pinning linux/amd64 unconditionally
# fails on an aarch64 host (an ARM-native base image has no amd64 variant to resolve), and an
# emulated run would make any timing meaningless. Set PLATFORM to override, e.g.
#   PLATFORM=linux/amd64 ./verify_gate.sh
PLATFORM_ARG=()
if [ -n "${PLATFORM:-}" ]; then PLATFORM_ARG=(--platform="$PLATFORM"); fi

echo "unit : $UNIT"
echo "base : $BASE_IMAGE"

docker build --quiet "${PLATFORM_ARG[@]}" -f "$HERE/Dockerfile.verify" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" -t "$TAG" "$HERE" >/dev/null
echo "built: $TAG"

IN="$(mktemp -d)"
OUT="$(mktemp -d)"
trap 'rm -rf "$IN" "$OUT"' EXIT
cp "$UNIT/scenario.json" "$IN/"

docker run --rm "${PLATFORM_ARG[@]}" --network=none \
    -v "$IN:/input:ro" -v "$OUT:/output" "$TAG" \
    simulate --config /input/scenario.json --out /output/trace.parquet >/dev/null

# Compare inside the image rather than on the host. The image already ships pandas and pyarrow, so
# the check needs nothing installed on the machine running this script, which is what makes it
# usable on a fresh box.
docker run --rm -i "${PLATFORM_ARG[@]}" --network=none \
    -v "$OUT:/out:ro" -v "$UNIT:/ref:ro" "$TAG" python - <<'PY'
import sys
from pathlib import Path

import pandas as pd

out, unit = Path("/out"), Path("/ref")
ok = True
for name in ("trace.parquet", "message_trace.parquet"):
    ref = unit / name
    if not ref.exists():
        continue
    got = pd.read_parquet(out / name)
    exp = pd.read_parquet(ref)
    same = got.equals(exp)
    ok &= same
    print(f"{name}: {'IDENTICAL' if same else 'DIFFERS'} ({len(got)} vs {len(exp)} rows)")
print("GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
