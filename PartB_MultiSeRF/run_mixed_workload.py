"""
Mixed-workload benchmark — closer to a system workload than single-selectivity
sweeps: one query stream mixing narrow and wide B windows, one shared
over-fetch setting per index (raised until mean recall >= 0.9 over the whole
stream), QPS measured over the stream.

Workload mix (fractions of the nq query stream):
  40% at s_B=1%, 20% at 5%, 20% at 10%, 10% at 25%, 10% at 50%

Arms: SeRF+ResidualB (K=1), Multi-SeRF K=4, Multi-SeRF K=16, and the adaptive
router (K=1 + K=16, tau=0.15). Same data/config as the main run
(n=5000, dim=32, nq=100, seed=0).

Usage: py -3 run_mixed_workload.py --out results_partB_mixed_workload.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import multiserf_proto as ms
from run_adaptive import adaptive_query_fn, cs_fn
from run_experiments_partB import (_clean_nan, gen_data, gen_queries,
                                   gen_ranges, run_exact_scan,
                                   sweep_qps_at_recall)

MIX = [(0.01, 0.40), (0.05, 0.20), (0.10, 0.20), (0.25, 0.10), (0.50, 0.10)]


def build_mixed_branges(nq: int) -> tuple[np.ndarray, list[float]]:
    """Per-query B windows drawn from the MIX distribution, deterministic.
    Reuses the same per-selectivity window seeds as the single-cell runs."""
    parts = []
    labels = []
    for s_B, frac in MIX:
        cnt = round(nq * frac)
        parts.append(gen_ranges(cnt, s_B, seed=1000 + int(s_B * 1e4))[:cnt])
        labels.extend([s_B] * cnt)
    return np.concatenate(parts, axis=0), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-build", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--out", type=str, default="results_partB_mixed_workload.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"== Mixed workload: n={args.n} dim={args.dim} nq={args.nq} "
          f"seed={args.seed} mix={MIX}")

    X, a, b = gen_data(args.n, args.dim, seed=args.seed)
    Q = gen_queries(args.nq, args.dim, seed=12345)
    branges, labels = build_mixed_branges(args.nq)
    assert branges.shape[0] == args.nq, "mix fractions must sum to 1"
    aranges = gen_ranges(args.nq, 1.0, seed=777)

    truths, _, avg_pass = run_exact_scan(X, a, b, Q, aranges, branges, args.k)
    print(f"   avg points passing both predicates over the stream: {avg_pass:.0f}")

    cs1 = ms.CompoundSegment(X, a, b, K=1, M=args.M, ef=args.ef_build)
    cs4 = ms.CompoundSegment(X, a, b, K=4, M=args.M, ef=args.ef_build)
    cs16 = ms.CompoundSegment(X, a, b, K=16, M=args.M, ef=args.ef_build)
    ai = ms.AdaptiveIndex(cs1, cs16, tau=args.tau)

    alpha_cap = max(2, args.n // args.k)
    alpha_grid = [x for x in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                  if x <= alpha_cap]

    routed: list = []
    arms = [("SeRF+ResidualB (K=1)", cs_fn(cs1)),
            ("Multi-SeRF K=4", cs_fn(cs4)),
            ("Multi-SeRF K=16", cs_fn(cs16)),
            (f"Adaptive (tau={args.tau})", adaptive_query_fn(ai, routed))]

    rows = {}
    base_qps = None
    for name, fn in arms:
        best, _ = sweep_qps_at_recall(fn, Q, aranges, branges, truths,
                                      args.k, alpha_grid)
        if base_qps is None:
            base_qps = best["qps"]
        rows[name] = {**best, "over_serf": best["qps"] / base_qps}
        print(f"   {name:<22} alpha={best['alpha']:>3} recall={best['recall']:.3f} "
              f"{best['qps']:7.1f} qps  ({rows[name]['over_serf']:.2f}x vs K=1)  "
              f"reached0.9={best.get('reached_target')}")
    rows[arms[-1][0]]["frac_bucketed"] = float(np.mean(routed)) if routed else 0.0

    payload = {"args": vars(args), "mix": MIX, "avg_pass_count": avg_pass,
               "arms": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(_clean_nan(payload), f, indent=2, allow_nan=False)
    print(f"Wrote {args.out}")

    print("\n| arm | alpha | recall | QPS | vs K=1 |")
    print("|---|---:|---:|---:|---:|")
    for name, v in rows.items():
        print(f"| {name} | {v['alpha']} | {v['recall']:.3f} "
              f"| {v['qps']:.1f} | {v['over_serf']:.2f}x |")


if __name__ == "__main__":
    main()
