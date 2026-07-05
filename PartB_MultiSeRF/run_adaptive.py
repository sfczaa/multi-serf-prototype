"""
Adaptive-routing experiment — does query-time arm selection get the best of
both K settings?

Builds the K=1 baseline and a K-bucket Compound Segment once, then sweeps
B-selectivity measuring QPS at recall >= 0.9 for three arms:

  - SeRF+ResidualB   (K=1, always)
  - Multi-SeRF       (K buckets, always)
  - Adaptive         (per query: estimated s_B <= tau -> bucketed, else K=1)

The interesting outcome is the ratio of each arm to SeRF+ResidualB: adaptive
should track Multi-SeRF where bucketing wins and fall back to ~1.0x where it
loses, resolving the K trade-off at query time instead of build time.

Usage: py -3 run_adaptive.py --n 5000 --dim 32 --nq 100 --K 16 --tau 0.15 \
           --out results_partB_adaptive.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

import multiserf_proto as ms
from run_experiments_partB import (_clean_nan, gen_data, gen_queries,
                                   gen_ranges, run_exact_scan,
                                   sweep_qps_at_recall)


def adaptive_query_fn(ai: ms.AdaptiveIndex, routed: list):
    """Adapter matching sweep_qps_at_recall's query_fn signature; records the
    routing decision per call into `routed`."""
    def fn(q, a_lo, a_hi, b_lo, b_hi, k, alpha):
        kp = max(k, int(math.ceil(alpha * k)))
        ef = max(kp, 64)
        ids, ds, vb, used_bucketed = ai.query(q, a_lo, a_hi, b_lo, b_hi, k,
                                              alpha=alpha, ef=ef)
        routed.append(used_bucketed)
        return ids, vb
    return fn


def cs_fn(cs):
    def fn(q, a_lo, a_hi, b_lo, b_hi, k, alpha):
        kp = max(k, int(math.ceil(alpha * k)))
        ef = max(kp, 64)
        ids, ds, vb = cs.query(q, a_lo, a_hi, b_lo, b_hi, k, alpha=alpha, ef=ef)
        return ids, vb
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-build", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.15,
                    help="route to the bucketed index when estimated s_B <= tau")
    ap.add_argument("--b-sweep", type=str, default="0.01,0.05,0.10,0.25,0.50")
    ap.add_argument("--out", type=str, default="results_partB_adaptive.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    b_sweep = [float(x) for x in args.b_sweep.split(",")]
    print(f"== Adaptive routing: n={args.n} dim={args.dim} nq={args.nq} "
          f"K={args.K} tau={args.tau} seed={args.seed}")

    X, a, b = gen_data(args.n, args.dim, seed=args.seed)
    Q = gen_queries(args.nq, args.dim, seed=12345)

    cs1 = ms.CompoundSegment(X, a, b, K=1, M=args.M, ef=args.ef_build)
    csK = ms.CompoundSegment(X, a, b, K=args.K, M=args.M, ef=args.ef_build)
    ai = ms.AdaptiveIndex(cs1, csK, tau=args.tau)
    print(f"index edges: K=1 {cs1.n_edges()}, K={args.K} {csK.n_edges()} "
          f"(both kept: ~{(cs1.n_edges() + csK.n_edges()) / cs1.n_edges():.2f}x "
          f"the single-graph edge count)")

    alpha_cap = max(2, args.n // args.k)
    alpha_grid = [x for x in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                  if x <= alpha_cap]

    results = []
    for s_B in b_sweep:
        branges = gen_ranges(args.nq, s_B, seed=1000 + int(s_B * 1e4))
        aranges = gen_ranges(args.nq, 1.0, seed=777)
        truths, _, avg_pass = run_exact_scan(X, a, b, Q, aranges, branges, args.k)

        serf_best, _ = sweep_qps_at_recall(
            cs_fn(cs1), Q, aranges, branges, truths, args.k, alpha_grid)
        cs_best, _ = sweep_qps_at_recall(
            cs_fn(csK), Q, aranges, branges, truths, args.k, alpha_grid)
        routed: list = []
        ad_best, _ = sweep_qps_at_recall(
            adaptive_query_fn(ai, routed), Q, aranges, branges, truths,
            args.k, alpha_grid)
        frac_bucketed = float(np.mean(routed)) if routed else 0.0

        row = {
            "s_B": s_B, "avg_pass_count": avg_pass,
            "serf_residualB": serf_best, "multiserf_cs": cs_best,
            "adaptive": ad_best,
            "adaptive_frac_bucketed": frac_bucketed,
            "cs_over_serf": cs_best["qps"] / serf_best["qps"],
            "adaptive_over_serf": ad_best["qps"] / serf_best["qps"],
        }
        results.append(row)
        print(f"-- s_B={s_B:.0%} (pass {avg_pass:.0f}): "
              f"serf={serf_best['qps']:.1f} cs={cs_best['qps']:.1f} "
              f"adaptive={ad_best['qps']:.1f} qps | "
              f"cs/serf={row['cs_over_serf']:.2f}x "
              f"adaptive/serf={row['adaptive_over_serf']:.2f}x "
              f"(routed to buckets: {frac_bucketed:.0%})")

    payload = {"args": vars(args), "results": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(_clean_nan(payload), f, indent=2, allow_nan=False)
    print(f"Wrote {args.out}")

    print("\n| s_B | CS/SeRF | Adaptive/SeRF | routed to buckets |")
    print("|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['s_B']:.0%} | {r['cs_over_serf']:.2f}x "
              f"| {r['adaptive_over_serf']:.2f}x "
              f"| {r['adaptive_frac_bucketed']:.0%} |")


if __name__ == "__main__":
    main()
