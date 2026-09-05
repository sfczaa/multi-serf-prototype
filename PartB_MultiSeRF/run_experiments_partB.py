"""
Part B experiment runner — Multi-SeRF Compound Segment vs baselines.

Tests the proposal's claim across a B-selectivity sweep, on the proposal's own
reporting axis: QPS at recall >= 0.9. Because the baselines reach recall 0.9
only by increasing over-fetch (costly), each ANN method sweeps its over-fetch
multiplier `alpha` until mean recall >= 0.9, and we report the QPS at that
(smallest, hence fastest) point. Exact methods (ExactScan, RangeFirstScan) are
recall 1.0 by construction; their latency is reported directly.

Methods
  - ExactScan        : full-table distance + both range filters (ground truth)
  - RangeFirstScan   : filter both predicates first, exact distance on survivors
  - HNSW+PostFilter  : plain ANN (full-A query on a K=1 graph), post-filter A & B
  - SeRF+ResidualB   : CompoundSegment K=1 (A in the graph, B residual filter)
  - Multi-SeRF (CS)  : CompoundSegment K>1 (B-bucket routing + A graph per bucket)

All graph methods share the SegmentGraph1D code (see design_notes.md §1.1), so
the Multi-SeRF vs SeRF+ResidualB delta isolates the B-bucketing contribution.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multiserf_proto as ms

_now = time.perf_counter


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------

def gen_data(n: int, dim: int, seed: int = 0, b_corr: float = 0.0):
    """Gaussian vectors + two Uniform[0,1] ordered attributes.

    b_corr > 0 makes B correlated with the vectors (Gaussian copula on the
    first vector coordinate, then rank-transformed back to a Uniform[0,1]
    marginal). Equal-frequency bucketing sees the same marginal either way;
    what changes is that B-buckets now map to regions of vector space — the
    stress case for bucket routing. b_corr = 0.0 leaves the RNG stream
    identical to the original recorded runs.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    a = rng.random(n)
    b = rng.random(n)
    if b_corr > 0.0:
        z = (b_corr * X[:, 0].astype(np.float64)
             + math.sqrt(1.0 - b_corr ** 2) * rng.standard_normal(n))
        b = (np.argsort(np.argsort(z)) + 0.5) / n
    return X, a, b


def gen_queries(nq: int, dim: int, seed: int = 12345) -> np.ndarray:
    """Held-out Gaussian query vectors (independent seed; not in the index)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nq, dim)).astype(np.float32)


def _read_fvecs(path: Path) -> np.ndarray:
    """TEXMEX .fvecs: per vector, an int32 dim followed by dim float32s."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    return np.ascontiguousarray(
        raw.reshape(-1, dim + 1)[:, 1:].view(np.float32))


SIFTSMALL_SHA256 = "b8f1e59b20319ac44279d5251706909dd3a5b8ca5ce2a11ddb1e73902252770e"


def load_siftsmall(seed: int = 0):
    """SIFT10K (TEXMEX `siftsmall`): 10k real 128-d SIFT base vectors plus the
    corpus's own 100 held-out query vectors. The corpus has no structured
    attributes, so a and b stay synthetic Uniform[0,1] — this run makes the
    *vectors* real, not the attribute distribution. Downloads ~5 MB into
    data/ on first use."""
    root = Path(__file__).resolve().parent / "data"
    d = root / "siftsmall"
    if not (d / "siftsmall_base.fvecs").exists():
        import hashlib
        import tarfile
        import urllib.request
        root.mkdir(exist_ok=True)
        tgz = root / "siftsmall.tar.gz"
        url = "https://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, tgz)
        # Pinned digest of the published TEXMEX archive (5,305,734 bytes,
        # 10,000 x 128-d base vectors). Verified before the archive is opened
        # so a corrupted or substituted download fails loudly here.
        digest = hashlib.sha256(tgz.read_bytes()).hexdigest()
        if digest != SIFTSMALL_SHA256:
            tgz.unlink()
            raise RuntimeError(
                f"{url} does not match the expected SHA-256. "
                f"expected {SIFTSMALL_SHA256}, got {digest}")
        with tarfile.open(tgz) as tf:
            root_resolved = root.resolve()
            for member in tf.getmembers():
                member_path = Path(member.name)
                target = (root / member_path).resolve()
                if target != root_resolved and root_resolved not in target.parents:
                    raise ValueError(f"unsafe archive member path: {member.name}")
                if member.issym() or member.islnk():
                    raise ValueError(f"archive links are not allowed: {member.name}")
                if not (member.isdir() or member.isfile()):
                    raise ValueError(f"unsupported archive member: {member.name}")
                tf.extract(member, root)
    X = _read_fvecs(d / "siftsmall_base.fvecs")
    Q = _read_fvecs(d / "siftsmall_query.fvecs")
    rng = np.random.default_rng(seed)
    a = rng.random(X.shape[0])
    b = rng.random(X.shape[0])
    return X, a, b, Q


def gen_ranges(nq: int, sel: float, seed: int) -> np.ndarray:
    """Per-query [lo, hi] windows of width `sel` over the Uniform[0,1] attribute
    domain, centred to avoid clipping (so realised selectivity ≈ sel)."""
    if sel >= 1.0:
        return np.tile(np.array([0.0, 1.0]), (nq, 1))
    rng = np.random.default_rng(seed)
    centers = rng.uniform(sel / 2.0, 1.0 - sel / 2.0, size=nq)
    return np.stack([centers - sel / 2.0, centers + sel / 2.0], axis=1)


def _clean_nan(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    return obj


# ----------------------------------------------------------------------
# exact references
# ----------------------------------------------------------------------

def run_exact_scan(X, a, b, Q, aranges, branges, k):
    """Full distance to all points, then filter. Latency reference; recall 1.0.
    Also returns ground-truth ids per query (reused by every ANN method)."""
    truths = []
    pass_counts = []
    t0 = _now()
    for i in range(Q.shape[0]):
        d = ms._l2_to_many(Q[i], X)
        mask = ((a >= aranges[i, 0]) & (a <= aranges[i, 1]) &
                (b >= branges[i, 0]) & (b <= branges[i, 1]))
        idx = np.nonzero(mask)[0]
        pass_counts.append(int(idx.size))
        if idx.size == 0:
            truths.append([])
            continue
        kk = min(k, idx.size)
        sub = d[idx]
        part = idx[np.argpartition(sub, kk - 1)[:kk]]
        part = part[np.argsort(d[part])]
        truths.append(part.tolist())
    qps = Q.shape[0] / (_now() - t0)
    return truths, qps, float(np.mean(pass_counts))


def run_range_first(X, a, b, Q, aranges, branges, k):
    """Filter both predicates first, exact distance on the survivors. Recall 1.0;
    latency scales with selectivity."""
    t0 = _now()
    for i in range(Q.shape[0]):
        mask = ((a >= aranges[i, 0]) & (a <= aranges[i, 1]) &
                (b >= branges[i, 0]) & (b <= branges[i, 1]))
        idx = np.nonzero(mask)[0]
        if idx.size:
            d = ms._l2_to_many(Q[i], X[idx])
            kk = min(k, idx.size)
            np.argpartition(d, kk - 1)[:kk]
    return Q.shape[0] / (_now() - t0)


# ----------------------------------------------------------------------
# ANN query closures (all share SegmentGraph1D via CompoundSegment)
# ----------------------------------------------------------------------

def cs_query_fn(cs, full_a):
    """Multi-SeRF / SeRF+ResidualB: route by B-bucket, A handled by the graph.
    `full_a` keeps the A range wide so SeRF+ResidualB does no A-filtering when
    the workload's A predicate is unrestricted."""
    def fn(q, a_lo, a_hi, b_lo, b_hi, k, alpha):
        kp = max(k, int(math.ceil(alpha * k)))
        ef = max(kp, 64)
        ids, ds, vb = cs.query(q, a_lo, a_hi, b_lo, b_hi, k, alpha=alpha, ef=ef)
        return ids, vb
    return fn


def hnsw_postfilter_fn(cs1):
    """Plain ANN over a single graph (full-A query), then post-filter A AND B.
    cs1 must be a K=1 CompoundSegment."""
    sg = cs1.buckets[0].sg
    bvals = cs1.b
    def fn(q, a_lo, a_hi, b_lo, b_hi, k, alpha):
        kp = max(k, int(math.ceil(alpha * k)))
        ef = max(kp, 64)
        ids, ds = sg.query(q, sg.a_min, sg.a_max, kp, ef)  # plain ANN (no A filt)
        # post-filter both predicates using global attribute arrays
        keep = [g for g in ids
                if (a_lo <= A_GLOBAL[g] <= a_hi) and (b_lo <= bvals[g] <= b_hi)]
        return keep[:k], 1
    return fn


def sweep_qps_at_recall(query_fn, Q, aranges, branges, truths, k,
                        alpha_grid, target=0.9):
    """Increase over-fetch alpha until mean recall >= target; return the QPS at
    that smallest alpha (= the fastest point that clears the recall bar), plus
    the full traced curve."""
    curve = []
    best = None
    for alpha in alpha_grid:
        t0 = _now()
        recs = []
        vbs = []
        for i in range(Q.shape[0]):
            ids, vb = query_fn(Q[i], aranges[i, 0], aranges[i, 1],
                               branges[i, 0], branges[i, 1], k, alpha)
            recs.append(ms.recall_at_k(ids, truths[i], k))
            vbs.append(vb)
        elapsed = _now() - t0
        rec = float(np.mean(recs))
        qps = Q.shape[0] / elapsed
        row = {"alpha": alpha, "recall": rec, "qps": qps,
               "mean_buckets": float(np.mean(vbs))}
        curve.append(row)
        if rec >= target:
            best = row
            break
    if best is None:                       # never reached target → report best
        best = max(curve, key=lambda r: r["recall"])
        best = {**best, "reached_target": False}
    else:
        best = {**best, "reached_target": True}
    return best, curve


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

A_GLOBAL = None  # set in main for the post-filter closure


def main():
    global A_GLOBAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--nq", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--K", type=int, default=16, help="bucket count for Multi-SeRF")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-build", type=int, default=64)
    ap.add_argument("--a-sel", type=float, default=1.0,
                    help="A-range selectivity (1.0 = full A range)")
    ap.add_argument("--b-corr", type=float, default=0.0,
                    help="correlation of attribute B with the vectors "
                         "(0.0 = independent, as in the original runs)")
    ap.add_argument("--dataset", choices=["synthetic", "siftsmall"],
                    default="synthetic",
                    help="siftsmall = real SIFT10K vectors + the corpus's 100 "
                         "queries; attributes stay synthetic")
    ap.add_argument("--b-sweep", type=str, default="0.01,0.05,0.10,0.25,0.50")
    ap.add_argument("--out", type=str, default="results_partB.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    b_sweep = [float(x) for x in args.b_sweep.split(",")]

    if args.dataset == "siftsmall":
        assert args.b_corr == 0.0, "--b-corr is defined for synthetic vectors only"
        X, a, b, Q = load_siftsmall(seed=args.seed)
        args.n, args.dim = int(X.shape[0]), int(X.shape[1])
        args.nq = min(args.nq, Q.shape[0])
        Q = Q[:args.nq]
        print(f"   siftsmall: n={args.n} dim={args.dim} nq={args.nq} "
              f"(real vectors, synthetic attributes)")
    else:
        X, a, b = gen_data(args.n, args.dim, seed=args.seed, b_corr=args.b_corr)
        Q = gen_queries(args.nq, args.dim, seed=12345)
    A_GLOBAL = a
    print(f"== Multi-SeRF experiment: n={args.n} dim={args.dim} nq={args.nq} "
          f"k={args.k} K={args.K} M={args.M} ef_build={args.ef_build} "
          f"a_sel={args.a_sel} b_corr={args.b_corr} seed={args.seed} "
          f"dataset={args.dataset}")

    # build the two indexes once
    print("building SeRF+ResidualB (K=1) ...")
    cs1 = ms.CompoundSegment(X, a, b, K=1, M=args.M, ef=args.ef_build, seed=args.seed)
    print(f"  build {cs1.build_s:.2f}s  edges={cs1.n_edges()}  "
          f"size~{cs1.size_bytes()/1e6:.2f} MB")
    print(f"building Multi-SeRF (K={args.K}) ...")
    csK = ms.CompoundSegment(X, a, b, K=args.K, M=args.M, ef=args.ef_build, seed=args.seed)
    print(f"  build {csK.build_s:.2f}s  edges={csK.n_edges()}  "
          f"size~{csK.size_bytes()/1e6:.2f} MB")

    # over-fetch grid (alpha = fetch / k), capped so kp <= n
    alpha_cap = max(2, args.n // args.k)
    alpha_grid = [a_ for a_ in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if a_ <= alpha_cap]

    fn_cs = cs_query_fn(csK, full_a=(args.a_sel >= 1.0))
    fn_serf = cs_query_fn(cs1, full_a=(args.a_sel >= 1.0))
    fn_hnsw = hnsw_postfilter_fn(cs1)

    results = []
    for s_B in b_sweep:
        branges = gen_ranges(args.nq, s_B, seed=1000 + int(s_B * 1e4))
        aranges = gen_ranges(args.nq, args.a_sel, seed=777)
        truths, exact_qps, avg_pass = run_exact_scan(X, a, b, Q, aranges, branges, args.k)
        rf_qps = run_range_first(X, a, b, Q, aranges, branges, args.k)
        avg_truth = float(np.mean([len(t) for t in truths]))

        serf_best, serf_curve = sweep_qps_at_recall(
            fn_serf, Q, aranges, branges, truths, args.k, alpha_grid)
        cs_best, cs_curve = sweep_qps_at_recall(
            fn_cs, Q, aranges, branges, truths, args.k, alpha_grid)
        hnsw_best, hnsw_curve = sweep_qps_at_recall(
            fn_hnsw, Q, aranges, branges, truths, args.k, alpha_grid)

        ratio = (cs_best["qps"] / serf_best["qps"]) if serf_best["qps"] else float("nan")
        row = {
            "s_B": s_B, "a_sel": args.a_sel,
            "avg_pass_count": avg_pass, "avg_truth_size": avg_truth,
            "exact_qps": exact_qps, "rangefirst_qps": rf_qps,
            "serf_residualB": serf_best, "multiserf_cs": cs_best,
            "hnsw_postfilter": hnsw_best,
            "cs_over_serf_qps_ratio": ratio,
            "serf_curve": serf_curve, "cs_curve": cs_curve,
        }
        results.append(row)
        print(f"\n-- s_B={s_B:.2%}  (avg {avg_pass:.0f} pts pass both predicates; "
              f"truth set {avg_truth:.0f})")
        print(f"   exact={exact_qps:7.1f} qps  range-first={rf_qps:7.1f} qps")
        print(f"   SeRF+ResidualB @recall>={serf_best['recall']:.3f}: "
              f"{serf_best['qps']:7.1f} qps (alpha={serf_best['alpha']}, "
              f"reached0.9={serf_best.get('reached_target')})")
        print(f"   Multi-SeRF     @recall>={cs_best['recall']:.3f}: "
              f"{cs_best['qps']:7.1f} qps (alpha={cs_best['alpha']}, "
              f"buckets={cs_best['mean_buckets']:.1f}/{args.K}, "
              f"reached0.9={cs_best.get('reached_target')})")
        print(f"   >>> CS / SeRF QPS ratio = {ratio:.2f}x")

    payload = {
        "args": vars(args),
        "index": {
            "serf_k1": {"build_s": cs1.build_s, "edges": cs1.n_edges(),
                        "size_bytes": cs1.size_bytes()},
            "multiserf_kK": {"build_s": csK.build_s, "edges": csK.n_edges(),
                             "size_bytes": csK.size_bytes(), "K": args.K},
        },
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(_clean_nan(payload), f, indent=2, allow_nan=False)
    print(f"\nWrote {args.out}")

    # summary table
    print("\n## Summary (QPS at recall >= 0.9; ratio = CS / SeRF+ResidualB)")
    print("| s_B | pts pass | SeRF+ResidualB qps (alpha) | Multi-SeRF qps (alpha, buckets) | ratio |")
    print("|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["serf_residualB"]; c = r["multiserf_cs"]
        print(f"| {r['s_B']:.0%} | {r['avg_pass_count']:.0f} "
              f"| {s['qps']:.1f} (a={s['alpha']}) "
              f"| {c['qps']:.1f} (a={c['alpha']}, {c['mean_buckets']:.1f}/{args.K}) "
              f"| {r['cs_over_serf_qps_ratio']:.2f}x |")


if __name__ == "__main__":
    main()
