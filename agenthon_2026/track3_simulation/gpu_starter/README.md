# GPU starter — one route to Track 3 throughput (not the expected one)

A minimal, honest scaffold for participants who want to try a GPU-accelerated port. **This is one
route, not the intended one.** Track 3 is a discrete-event simulation with exact-fill gates; the
event loop is inherently sequential, and a naive GPU/batched port breaks exactness and is rejected.
CPU-optimized submissions are first-class — nothing here is required, and there is no GPU mandate.

## The one discipline that matters

Every GPU result must be **byte-identical** to the CPU reference, or the Tier-A gate rejects the run.
[`gpu_kernel_example.py`](gpu_kernel_example.py) shows the verification pattern and runs anywhere
(CuPy on a GPU, NumPy as a CPU fallback):

```bash
python gpu_kernel_example.py
# backend: NumPy (CPU fallback)   (or "CuPy (GPU)" on a GPU box)
# bit-identical to CPU reference: True
```

## What is and isn't safe to accelerate

- **Safe:** element-wise **integer / comparison** work on independent elements (prices, sizes,
  order_ids are integers) — associative, order-independent, bit-exact. Also legitimately parallel:
  running the *independent* scenarios of the batch family concurrently (each still reproduced
  exactly — see the batch isolation gate).
- **Not safe on the exact path:** float reductions (last-bit nondeterminism), unstable sorts /
  argsort (GPU tie-breaking differs from CPU and reorders equal-priority events), atomics without a
  fixed reduction order. Keep these off the ranked path or reproduce the CPU order explicitly.
- **Stays on CPU:** the event queue and matching *order* — sequential and order-dependent, so it
  can't be vectorized without changing execution order.

## Proof that the starter is gate-compatible

`verify_gate.sh` builds the starter onto the Track 3 baseline and checks it still reproduces a real
public unit **bit-identically**, so you can see that adding the accelerated helpers and the
self-checking shim does not perturb the simulation:

Build the Track 3 baseline first — `verify_gate.sh` layers the starter onto it:

```bash
docker build --platform=linux/amd64 -t track3-abides-baseline:latest ..   # from baselines/gpu_starter/
./verify_gate.sh                       # defaults to units/t3-eq-deterministic-baseline
./verify_gate.sh ../../units/t3-s001-price-time-priority
./verify_gate.sh ../../units/t3-s001-price-time-priority my-own-base:tag   # second arg overrides the base
```

Current result on `t3-eq-deterministic-baseline`:

```
trace.parquet: IDENTICAL (10669 vs 10669 rows)
message_trace.parquet: IDENTICAL (18568 vs 18568 rows)
GATE: PASS
```

No GPU is required: this exercises the CuPy-absent fallback, which is exactly the point.

The CUDA path itself is now verified too. On an NVIDIA GH200 (CUDA 13, aarch64) the example runs with
real CuPy and reports:

```
backend: CuPy (GPU)
bit-identical to CPU reference: True
```

so the element-wise integer work this file demonstrates stays bit-exact between CPU and GPU on real
hardware, which is the whole premise of an admissible GPU port. The image's
`simulate` shim runs the exactness self-check first and exits non-zero if the accelerated helpers
ever stop matching the CPU reference, rather than emitting a trace the Tier-A gate would reject.

## Building (`network=none`-compatible)

The runtime forbids network, so the CUDA runtime and all wheels are vendored at **build** time:

```bash
docker build -f Dockerfile -t track3-gpu-starter:dev .
```

The [`Dockerfile`](Dockerfile) starts from an NVIDIA CUDA **`-runtime`** base, installs CuPy + the
data stack, points `CUDA_PATH` at CuPy's vendored headers, and runs the exactness self-check at
build time.

**Why `CUDA_PATH`.** CuPy JIT-compiles its element-wise kernels through nvrtc the first time
they run, so it needs real CUDA headers at *run* time. A `-runtime` base does not ship them — but
CuPy vendors its own copies, and the only reason nvrtc cannot see them is that the `-I` flags CuPy
passes are built from `CUDA_PATH`. Point `CUDA_PATH` at a directory containing them and the small
base works.

Measured on a B200 (NVIDIA, 2026-08-19; records in the organizer runner repo,
`verification/2026-08-19-gpu-starter-b200/` and `.../-runtime-base/`), all under gVisor:

| variant | result | image |
|---|---|---|
| `-runtime` as shipped | `cuda_fp16.h` not found, exit 1 | 4.02 GB |
| `-runtime` + `nvidia-cuda-runtime-cu12` | **still** not found, exit 1 | 4.03 GB |
| `-runtime` + `CUDA_PATH` | `backend: CuPy (GPU)`, bit-identical, exit 0 | 4.02 GB |
| `-devel` | `backend: CuPy (GPU)`, bit-identical, exit 0 | **10 GB** |

Two traps this corrected:

- **`cupy-cudaXXx[ctk]` does not fix it.** On `linux/amd64` with `cupy-cuda12x` 13.6.0 the extra
  installs *nothing* — no `nvidia-*` packages at all — and pip does not fail on an extra it does
  not recognise, so it fails silently and late. Earlier advice here came from a GH200 run (CUDA 13,
  aarch64), a different wheel, and did not transfer.
- **`nvidia-cuda-runtime-cu12` looks like it helps and does not.** It installs the header; CuPy
  still cannot find it, because CuPy does not search pip's `nvidia/` tree.

**Limit — when to use `-devel` instead.** CuPy's vendored headers are a *subset* of a full CTK.
They cover this element-wise example; a submission reaching for cuBLAS, cuRAND or thrust may still
fail to compile, and that is **untested**. If you hit it, switch the base to
`nvidia/cuda:12.8.0-devel-ubuntu24.04` and drop the `CUDA_PATH` block. That costs about **6 GB**
(10 GB against 4 GB, measured — this file previously estimated ~3 GB, before anyone measured it),
which counts against the image-size guidance in [`../README.md`](../README.md).

**Good news for anyone targeting Blackwell:** the same run confirmed CuPy JITs correctly for
`sm_100` (compute capability 10.0) and that the whole nvrtc path works under gVisor, producing
bit-identical output. There is no wheel-versus-Blackwell problem to design around.

For a full submission, fetch the ABIDES adapter exactly as `baselines/Dockerfile` does and
implement the `simulate` verb on top.

## `TODO(hub)` — needed to make this real on the eval box

- Pin the CUDA base image + `cupy-cudaXXX` to the box's **CUDA toolkit / driver**, and target its
  **GPU SKU + compute capability** (`baselines/README.md` §5 lists these once the hub confirms them).
- Validate a gate-passing GPU port on the **actual box GPU** — a bit-exact GPU matching engine is a
  research effort and can't be certified without the hardware. Until then this is a *scaffold + the
  exactness discipline*, not a finished accelerated simulator.

## How the awards see a GPU submission

- **Best GPU Acceleration** ranks on host-measured GPU efficiency — the harness measures your GPU
  time (NVML), you don't self-report it (`throughput/README.md`).
- Pair it with an Nsight SimProfile for **Best Systems Diagnosis** (`docs/PROFILING.md`).
- The NVIDIA-stack mapping for T3 is in `docs/NVIDIA-STACK.md`.
