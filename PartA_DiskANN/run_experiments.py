"""
Experiment runner for the Part A prototype.

Runs:
  - exact numpy scan        (ground truth, baseline 1)
  - pynndescent in-memory   (graph-based ANN baseline 2; HNSW stand-in)
  - proto-eager             (our index, all segments loaded)
  - proto-mmap              (our index, segments mmap'd)

Prints a markdown table and writes raw JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# allow running from this dir
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diskann_proto as dap

# Use perf_counter for all timing — monotonic, higher resolution than time.time.
_now = time.perf_counter

try:
    import psutil  # type: ignore
    _PROC = psutil.Process()
    def rss_mb() -> float:
        return _PROC.memory_info().rss / (1024 * 1024)
except Exception:
    _PROC = None
    def rss_mb() -> float:
        return float("nan")


def gen_gaussian(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def gen_held_out_queries(nq: int, dim: int, seed: int = 12345) -> np.ndarray:
    """Generate a held-out Gaussian query set.

    Drawn from the same N(0, I) distribution as the index data, but with an
    independent seed, so query vectors are (with probability ~1) NOT present
    in the index. Sampling queries from X itself would let the index trivially
    return the exact match as the top-1, inflating recall.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nq, dim)).astype(np.float32)


def load_fvecs(path: str) -> np.ndarray:
    """Load a TEXMEX `.fvecs` file (SIFT, GIST, etc.).

    Format: each vector is `int32 dim` followed by `float32 dim` values,
    concatenated for all vectors. Returns an (n, dim) float32 array.
    """
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path}: empty file")
    dim = int(raw[0])
    if dim <= 0 or dim > 100_000:
        raise ValueError(f"{path}: implausible dim {dim} — wrong format?")
    rec_len = dim + 1
    if raw.size % rec_len != 0:
        raise ValueError(
            f"{path}: size {raw.size} not a multiple of rec_len {rec_len}")
    n = raw.size // rec_len
    arr = raw.reshape(n, rec_len)
    if not (arr[:, 0] == dim).all():
        raise ValueError(f"{path}: inconsistent per-record dim header")
    return np.ascontiguousarray(arr[:, 1:].view(np.float32))


def load_sift(sift_dir: str, max_n: int | None = None,
              max_nq: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load SIFT (or any TEXMEX-format) base + query vectors from `sift_dir`.

    Expects, in order of preference, one of these filename pairs in the dir:
      - siftsmall_base.fvecs / siftsmall_query.fvecs   (10k base / 100 query)
      - sift_base.fvecs      / sift_query.fvecs        (1M base / 10k query)

    Raises FileNotFoundError if neither pair is present, which the caller can
    catch to fall back to synthetic Gaussian. `max_n` / `max_nq` truncate the
    loaded arrays (no shuffling — keeps loads deterministic).
    """
    d = Path(sift_dir)
    candidates = [
        ("siftsmall_base.fvecs", "siftsmall_query.fvecs"),
        ("sift_base.fvecs",      "sift_query.fvecs"),
    ]
    for base_name, query_name in candidates:
        bp, qp = d / base_name, d / query_name
        if bp.exists() and qp.exists():
            X = load_fvecs(str(bp))
            Q = load_fvecs(str(qp))
            if max_n is not None and max_n < X.shape[0]:
                X = X[:max_n]
            if max_nq is not None and max_nq < Q.shape[0]:
                Q = Q[:max_nq]
            return X, Q
    raise FileNotFoundError(
        f"no fvecs pair found in {sift_dir} "
        f"(looked for {[c[0] for c in candidates]})")


def _clean_nan(obj):
    """Recursively replace float NaN/Inf with None so the JSON output is
    valid (vanilla json.dump emits the literal `NaN`, which is not valid JSON
    and trips strict parsers / reviewers)."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    return obj


def percentiles(xs: list[float], qs=(50, 95, 99)) -> dict:
    a = np.array(xs, dtype=np.float64)
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def bench_exact(X: np.ndarray, Q: np.ndarray, k: int) -> dict:
    t0 = _now()
    truth = dap.exact_knn(X, Q, k)
    elapsed = _now() - t0
    return {
        "name": "exact",
        "build_s": 0.0,
        "query_total_s": elapsed,
        "query_mean_ms": elapsed * 1000 / Q.shape[0],
        "recall@10": 1.0,
        "ids": truth.tolist(),
    }


def bench_pynndescent(X: np.ndarray, Q: np.ndarray, k: int, truth: np.ndarray,
                      M: int) -> dict:
    try:
        from pynndescent import NNDescent
    except Exception as e:
        return {"name": "pynndescent", "error": str(e)}
    rss_before = rss_mb()
    t0 = _now()
    index = NNDescent(X, n_neighbors=max(30, k * 3), metric="euclidean",
                      random_state=0, low_memory=False)
    index.prepare()
    build_s = _now() - t0
    rss_after_build = rss_mb()
    # warmup
    _ = index.query(Q[:1], k=k)
    lats = []
    t_all = _now()
    pred_ids = np.zeros((Q.shape[0], k), dtype=np.int64)
    for i in range(Q.shape[0]):
        t_q = _now()
        ids, _ = index.query(Q[i:i + 1], k=k)
        lats.append((_now() - t_q) * 1000)
        pred_ids[i] = ids[0]
    total_s = _now() - t_all
    rec = dap.recall_at_k(pred_ids, truth, k)
    return {
        "name": "pynndescent",
        "build_s": build_s,
        "query_total_s": total_s,
        "query_mean_ms": total_s * 1000 / Q.shape[0],
        "recall@10": rec,
        "rss_before_mb": rss_before,
        "rss_after_build_mb": rss_after_build,
        **percentiles(lats),
    }


def bench_proto(X: np.ndarray, Q: np.ndarray, k: int, truth: np.ndarray,
                index_path: str, M: int, ef_construction: int, pq_m: int,
                L: int, mode: str, label: str | None = None) -> dict:
    name = label if label is not None else f"proto-{mode}-L{L}"
    if not Path(index_path).exists():
        rss_before = rss_mb()
        t0 = _now()
        stats = dap.build_and_write(
            X, index_path,
            M=M, ef_construction=ef_construction, pq_m=pq_m,
            verbose=False,
        )
        build_s = _now() - t0
        rss_after_build = rss_mb()
    else:
        stats = {}
        build_s = float("nan")
        rss_before = rss_mb()
        rss_after_build = rss_mb()

    reader = dap.IndexReader(index_path, mode=mode)
    rss_after_open = rss_mb()
    # warmup
    _ = reader.beam_search(Q[0], k=k, L=L)
    lats = []
    pred_ids = np.zeros((Q.shape[0], k), dtype=np.int64)
    t_all = _now()
    for i in range(Q.shape[0]):
        t_q = _now()
        ids, _ = reader.beam_search(Q[i], k=k, L=L)
        lats.append((_now() - t_q) * 1000)
        pred_ids[i, :len(ids)] = ids
    total_s = _now() - t_all
    reader.close()
    rec = dap.recall_at_k(pred_ids, truth, k)
    return {
        "name": name,
        "build_s": build_s,
        "query_total_s": total_s,
        "query_mean_ms": total_s * 1000 / Q.shape[0],
        "recall@10": rec,
        "rss_before_mb": rss_before,
        "rss_after_build_mb": rss_after_build,
        "rss_after_open_mb": rss_after_open,
        "L": L,
        "file_bytes": Path(index_path).stat().st_size,
        "layout": stats,
        **percentiles(lats),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=96)
    ap.add_argument("--pq-m", type=int, default=16)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--index", type=str, default="proto.idx")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--skip-pynnd", action="store_true")
    ap.add_argument("--dataset", type=str, default="gaussian",
                    choices=["gaussian", "sift"],
                    help="gaussian (synthetic) or sift (.fvecs files)")
    ap.add_argument("--sift-dir", type=str, default="siftsmall",
                    help="directory containing siftsmall_base.fvecs etc.")
    args = ap.parse_args()

    np.random.seed(0)
    if args.dataset == "sift":
        try:
            X, Q = load_sift(args.sift_dir, max_n=args.n, max_nq=args.nq)
            print(f"== dataset: SIFT (from {args.sift_dir}) "
                  f"n={X.shape[0]} dim={X.shape[1]} nq={Q.shape[0]} k={args.k}")
            args.dim = X.shape[1]
        except FileNotFoundError as e:
            print(f"WARN: {e}; falling back to gaussian")
            args.dataset = "gaussian"
    if args.dataset == "gaussian":
        print(f"== dataset: gaussian n={args.n} dim={args.dim} "
              f"nq={args.nq} k={args.k}")
        X = gen_gaussian(args.n, args.dim, seed=0)
        # Held-out query set: independent seed from X. Sampling Q from X would
        # let the index return the exact match as top-1, which inflates recall
        # and is not representative of real "unseen query" workloads.
        Q = gen_held_out_queries(args.nq, args.dim, seed=12345)
    print(f"  X.nbytes = {X.nbytes/1e6:.1f} MB  Q.nbytes = {Q.nbytes/1e6:.2f} MB")

    if args.rebuild and Path(args.index).exists():
        Path(args.index).unlink()

    results = []

    print("\n== exact (ground truth)")
    r = bench_exact(X, Q, args.k)
    truth = np.array(r.pop("ids"))
    print(f"  mean query: {r['query_mean_ms']:.2f} ms")
    results.append(r)

    if not args.skip_pynnd:
        print("\n== pynndescent (in-memory graph baseline)")
        try:
            r = bench_pynndescent(X, Q, args.k, truth, args.M)
            print(f"  build: {r.get('build_s', float('nan')):.2f}s  "
                  f"mean q: {r.get('query_mean_ms', float('nan')):.2f} ms  "
                  f"recall@10: {r.get('recall@10', float('nan')):.3f}")
            results.append(r)
        except Exception as e:
            print(f"  failed: {e}")

    # sweep beam width L for the prototype, in both eager and mmap modes
    L_sweep = sorted(set([args.L // 2 if args.L >= 32 else args.L, args.L, args.L * 2]))
    first_eager = True
    for L in L_sweep:
        print(f"\n== proto-eager L={L}")
        r = bench_proto(X, Q, args.k, truth, args.index, args.M,
                        args.ef_construction, args.pq_m, L, mode="eager",
                        label=f"proto-eager-L{L}")
        if first_eager:
            print(f"  build: {r['build_s']:.2f}s  file size: {r['file_bytes']/1e6:.2f} MB")
            first_eager = False
        print(f"  mean q: {r['query_mean_ms']:.2f} ms  recall@10: {r['recall@10']:.3f}")
        results.append(r)
    for L in L_sweep:
        print(f"\n== proto-mmap L={L}")
        r = bench_proto(X, Q, args.k, truth, args.index, args.M,
                        args.ef_construction, args.pq_m, L, mode="mmap",
                        label=f"proto-mmap-L{L}")
        print(f"  mean q: {r['query_mean_ms']:.2f} ms  recall@10: {r['recall@10']:.3f}")
        results.append(r)

    payload = _clean_nan({"args": vars(args), "results": results})
    with open(args.out, "w", encoding="utf-8") as f:
        # allow_nan=False would raise on stray NaN; _clean_nan already replaced
        # them with null, so we set it for defence-in-depth.
        json.dump(payload, f, indent=2, allow_nan=False)
    print(f"\nWrote {args.out}")

    # markdown table
    print("\n## Summary")
    print("| method | build (s) | mean q (ms) | p95 (ms) | p99 (ms) | recall@10 |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in results:
        bs = f"{r.get('build_s', float('nan')):.2f}"
        mq = f"{r.get('query_mean_ms', float('nan')):.2f}"
        p95 = f"{r.get('p95', float('nan')):.2f}"
        p99 = f"{r.get('p99', float('nan')):.2f}"
        rec = f"{r.get('recall@10', float('nan')):.3f}"
        print(f"| {r['name']} | {bs} | {mq} | {p95} | {p99} | {rec} |")


if __name__ == "__main__":
    main()
