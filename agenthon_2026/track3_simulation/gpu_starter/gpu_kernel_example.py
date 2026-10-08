"""Illustrative GPU-acceleration starter for Track 3 — ONE route, not the expected one.

The only discipline that matters for a Tier-A GPU port: every GPU result must be BYTE-IDENTICAL to
the CPU reference, because the exact-fill gate rejects any divergence. This example uses CuPy on a
GPU and falls back to NumPy on CPU, so it runs anywhere; the point is the *verification pattern*, not
a full simulator.

Why the discrete-event loop itself does NOT belong on the GPU: it is sequential and order-dependent
(each event can depend on every prior event through the shared order book and the message queue), so
vectorizing or batching the matching order changes execution order and breaks exactness. What IS
safe to accelerate is per-event work on independent elements using **integer / comparison** ops —
prices, sizes, and order_ids are integers in the trace schema, so element-wise integer math is
associative, order-independent, and bit-exact on both CPU and GPU.

The usual bit-exactness breakers, to avoid on the ranked path:
  * float reductions (sum/mean over a GPU array) — non-deterministic ordering changes the last bits;
  * unstable sorts / argsort — GPU tie-breaking differs from CPU, reordering equal-priority events;
  * atomics without a fixed reduction order.
Keep those off the exact path, or reproduce the CPU order explicitly.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as xp  # type: ignore

    # Probe, do not merely import. CuPy initialises lazily, so `import cupy` succeeds on a host
    # with no usable driver and the failure surfaces at the FIRST array operation instead —
    # outside this `try`, where the NumPy fallback below can no longer catch it. Measured on a
    # B200 host with no GPU attached to the build step:
    #     python3 -c 'import cupy'                    -> OK, no driver needed
    #     python3 -c 'import cupy; cupy.asarray([1])' -> CUDARuntimeError: cudaErrorInsufficientDriver
    # Without this line the build-time self-check below cannot pass on any stock Docker daemon.
    xp.asarray([0])
    _ON_GPU = True
except Exception:
    xp = np  # NumPy fallback so this file runs without a GPU
    _ON_GPU = False


def _to_host(a: object) -> np.ndarray:
    """Bring an array back to host NumPy for exact comparison (no-op on the CPU path)."""
    return xp.asnumpy(a) if _ON_GPU else np.asarray(a)


def per_order_notional(price: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Element-wise integer notional = price × size. Order-independent and bit-exact."""
    p = xp.asarray(price, dtype=xp.int64)
    s = xp.asarray(size, dtype=xp.int64)
    return _to_host(p * s)


def marketable_mask(
    order_price: np.ndarray, order_is_bid: np.ndarray, best_bid: int, best_ask: int
) -> np.ndarray:
    """Which incoming orders would cross the touch: a bid at/above best_ask, or an ask at/below
    best_bid. Pure element-wise integer comparison — exact on CPU and GPU."""
    px = xp.asarray(order_price, dtype=xp.int64)
    is_bid = xp.asarray(order_is_bid, dtype=xp.bool_)
    crosses = xp.where(is_bid, px >= best_ask, px <= best_bid)
    return _to_host(crosses)


def _cpu_reference(price: np.ndarray, size: np.ndarray, is_bid: np.ndarray) -> tuple:
    notional = price.astype(np.int64) * size.astype(np.int64)
    crosses = np.where(is_bid, price >= 100_010, price <= 100_000)
    return notional, crosses


def main() -> int:
    rng = np.random.default_rng(0)
    n = 100_000
    price = rng.integers(99_990, 100_020, size=n, dtype=np.int64)
    size = rng.integers(1, 50, size=n, dtype=np.int64)
    is_bid = rng.integers(0, 2, size=n, dtype=np.int64).astype(bool)

    gpu_notional = per_order_notional(price, size)
    gpu_crosses = marketable_mask(price, is_bid, best_bid=100_000, best_ask=100_010)
    cpu_notional, cpu_crosses = _cpu_reference(price, size, is_bid)

    ok = np.array_equal(gpu_notional, cpu_notional) and np.array_equal(
        gpu_crosses, cpu_crosses
    )
    backend = "CuPy (GPU)" if _ON_GPU else "NumPy (CPU fallback)"
    print(f"backend: {backend}")
    print(f"bit-identical to CPU reference: {ok}")
    if not ok:
        print("EXACTNESS BROKEN — this port would fail the Tier-A gate")
        return 1
    print("OK — this integer/comparison work is safe to accelerate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
