"""
Quick demo — build a small Multi-SeRF index and watch bucket routing work.

Runs in ~20-30 s:  py -3 demo.py

Builds the K=1 baseline (SeRF+ResidualB) and a K=16 Compound Segment over the
same 5,000 synthetic points, then answers the same filtered-ANN workload with
both, plus the adaptive router. For each B-window width it reports the
over-fetch each arm needs to reach recall 0.9, the buckets it searched, and
the throughput. Output is deterministic apart from the timings.
"""

from __future__ import annotations

import math
import time

import numpy as np

import multiserf_proto as ms
from run_experiments_partB import (gen_data, gen_queries, gen_ranges,
                                   run_exact_scan, sweep_qps_at_recall)

N, DIM, NQ, K, KNN = 5000, 32, 30, 16, 10


def arm_fn(index):
    def fn(q, a_lo, a_hi, b_lo, b_hi, k, alpha):
        kp = max(k, int(math.ceil(alpha * k)))
        ef = max(kp, 64)
        out = index.query(q, a_lo, a_hi, b_lo, b_hi, k, alpha=alpha, ef=ef)
        return out[0], out[2]        # ids, buckets_visited
    return fn


def main():
    print("Multi-SeRF demo: filtered ANN = vector search + two range predicates")
    print(f"data: n={N} synthetic Gaussian vectors (dim={DIM}), "
          f"attributes a,b ~ Uniform[0,1]\n")

    X, a, b = gen_data(N, DIM, seed=0)
    Q = gen_queries(NQ, DIM, seed=12345)

    t0 = time.perf_counter()
    cs1 = ms.CompoundSegment(X, a, b, K=1)
    t1 = time.perf_counter()
    csK = ms.CompoundSegment(X, a, b, K=K)
    t2 = time.perf_counter()
    adaptive = ms.AdaptiveIndex(cs1, csK, tau=0.15)
    print(f"built SeRF+ResidualB (K=1):  {t1 - t0:5.1f}s, {cs1.n_edges()} edges")
    print(f"built Multi-SeRF   (K={K}): {t2 - t1:5.1f}s, {csK.n_edges()} edges "
          f"(bucketing is ~free in space and builds faster)\n")

    alpha_grid = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    arms = [("SeRF+ResidualB", arm_fn(cs1)),
            (f"Multi-SeRF K={K}", arm_fn(csK)),
            ("Adaptive router", arm_fn(adaptive))]

    for s_B, story in [(0.01, "narrow B window -> bucket routing shines"),
                       (0.10, "medium window -> still ahead"),
                       (0.50, "wide window -> single graph is the right tool")]:
        print(f"== B selectivity {s_B:.0%}: {story}")
        branges = gen_ranges(NQ, s_B, seed=1000 + int(s_B * 1e4))
        aranges = gen_ranges(NQ, 1.0, seed=777)
        truths, _, avg_pass = run_exact_scan(X, a, b, Q, aranges, branges, KNN)
        print(f"   ({avg_pass:.0f} of {N} points pass both predicates on average)")
        base_qps = None
        for name, fn in arms:
            best, _ = sweep_qps_at_recall(fn, Q, aranges, branges, truths,
                                          KNN, alpha_grid)
            note = f"searched {best['mean_buckets']:.1f} bucket(s)"
            if name.startswith("SeRF"):
                base_qps = best["qps"]
                note = "single graph + residual-B filter"
            rel = f"{best['qps'] / base_qps:4.1f}x" if base_qps else "  --"
            print(f"   {name:<16} alpha={best['alpha']:>3} -> "
                  f"recall {best['recall']:.2f}, {best['qps']:7.1f} qps "
                  f"({rel} vs baseline; {note})")
        print()

    print("Takeaway: routing by the B predicate wins when it is selective, and")
    print("the adaptive router falls back to the single graph when it is not.")
    print("Full experiments: run_experiments_partB.py / run_adaptive.py;")
    print("recorded results and analysis: results.md")


if __name__ == "__main__":
    main()
