"""
Adaptive-routing threshold sensitivity — is the §10 result specific to τ=0.15?

Same main config as run_adaptive.py (n=5000, dim=32, nq=100, K=16, seed=0),
but the adaptive arm is measured at several routing thresholds τ. The K=1 and
K=16 arms do not depend on τ, so they are measured once per B-selectivity and
every adaptive/τ cell is reported against the same SeRF+ResidualB baseline.

Usage: py -3 run_adaptive_tau_sweep.py --out results_partB_adaptive_tau.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-build", type=int, default=64)
    ap.add_argument("--tau-sweep", type=str, default="0.05,0.10,0.15,0.25,0.50")
    ap.add_argument("--b-sweep", type=str, default="0.01,0.05,0.10,0.25,0.50")
    ap.add_argument("--out", type=str, default="results_partB_adaptive_tau.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    taus = [float(x) for x in args.tau_sweep.split(",")]
    b_sweep = [float(x) for x in args.b_sweep.split(",")]
    print(f"== Adaptive tau sensitivity: n={args.n} dim={args.dim} nq={args.nq} "
          f"K={args.K} seed={args.seed} taus={taus}")

    X, a, b = gen_data(args.n, args.dim, seed=args.seed)
    Q = gen_queries(args.nq, args.dim, seed=12345)
    cs1 = ms.CompoundSegment(X, a, b, K=1, M=args.M, ef=args.ef_build)
    csK = ms.CompoundSegment(X, a, b, K=args.K, M=args.M, ef=args.ef_build)

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

        by_tau = {}
        for tau in taus:
            ai = ms.AdaptiveIndex(cs1, csK, tau=tau)
            routed: list = []
            ad_best, _ = sweep_qps_at_recall(
                adaptive_query_fn(ai, routed), Q, aranges, branges, truths,
                args.k, alpha_grid)
            by_tau[str(tau)] = {
                **ad_best,
                "frac_bucketed": float(np.mean(routed)) if routed else 0.0,
                "over_serf": ad_best["qps"] / serf_best["qps"],
            }

        results.append({
            "s_B": s_B, "avg_pass_count": avg_pass,
            "serf_residualB": serf_best, "multiserf_cs": cs_best,
            "cs_over_serf": cs_best["qps"] / serf_best["qps"],
            "adaptive_by_tau": by_tau,
        })
        cells = "  ".join(f"tau={t}: {v['over_serf']:.2f}x({v['frac_bucketed']:.0%})"
                          for t, v in by_tau.items())
        print(f"-- s_B={s_B:.0%}: cs/serf={results[-1]['cs_over_serf']:.2f}x | {cells}")

    payload = {"args": vars(args), "results": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(_clean_nan(payload), f, indent=2, allow_nan=False)
    print(f"Wrote {args.out}")

    print("\n| s_B | CS/SeRF |" + "".join(f" tau={t} |" for t in taus))
    print("|---:|---:|" + "---:|" * len(taus))
    for r in results:
        row = f"| {r['s_B']:.0%} | {r['cs_over_serf']:.2f}x |"
        for t in taus:
            v = r["adaptive_by_tau"][str(t)]
            row += f" {v['over_serf']:.2f}x ({v['frac_bucketed']:.0%}) |"
        print(row)


if __name__ == "__main__":
    main()
