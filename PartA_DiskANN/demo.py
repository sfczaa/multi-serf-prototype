"""
Quick demo — build a small on-disk index and watch the moving parts.

Runs in ~30-60 s:  py -3 demo.py

Builds a Vamana graph + PQ codes over 1,000 synthetic vectors, writes the
page-aligned index file, reads it back in both access modes, and reports
recall / latency / file size. The index file is written to a temp directory
and deleted afterwards. Deterministic apart from the timings.
"""

from __future__ import annotations

import struct
import tempfile
import time
from pathlib import Path

import numpy as np

import diskann_proto as dap

N, DIM, NQ, K = 1000, 64, 30, 10
M, EF, PQ_M, PQ_KS = 16, 48, 8, 64
_now = time.perf_counter


def main():
    print("Part A demo: DiskANN-style on-disk ANN index")
    print(f"data: n={N} synthetic Gaussian vectors (dim={DIM}); "
          f"{NQ} held-out queries\n")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, DIM)).astype(np.float32)
    Q = np.random.default_rng(12345).standard_normal((NQ, DIM)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "demo.idx")
        print(f"building (Vamana M={M} ef={EF} two-pass + PQ m={PQ_M} ks={PQ_KS}) ...")
        stats = dap.build_and_write(X, path, M=M, ef_construction=EF,
                                    pq_m=PQ_M, pq_ks=PQ_KS)
        print(f"  PQ train {stats['t_pq_train']:.1f}s + encode "
              f"{stats['t_pq_encode']:.1f}s + graph {stats['t_build_graph']:.1f}s "
              f"+ write {stats['t_write']:.2f}s = {stats['build_time_total']:.1f}s")

        raw = N * (4 + 4 * M + PQ_M + 4 * DIM)
        print(f"\nfile: {stats['total_bytes'] / 1e6:.2f} MB total "
              f"(raw payload {raw / 1e6:.2f} MB + page/record padding)")
        print(f"  header 4 KB | graph {stats['graph_bytes'] / 1e6:.2f} MB "
              f"({stats['rec_size']} B/node, {stats['nodes_per_page']} nodes/page) "
              f"| PQ codes {stats['pq_bytes'] / 1e6:.2f} MB "
              f"| full vectors {stats['full_bytes'] / 1e6:.2f} MB "
              f"| codebook {stats['codebook_bytes'] / 1e6:.2f} MB")

        with open(path, "rb") as f:
            hdr = struct.unpack_from(dap.HEADER_STRUCT, f.read(dap.HEADER_SIZE), 0)
        print(f"  header readback: magic={hdr[0]:#x} version={hdr[1]} "
              f"n={hdr[2]} dim={hdr[3]} M={hdr[4]} entry={hdr[5]}")

        truth = dap.exact_knn(X, Q, K)
        print(f"\nquerying ({NQ} held-out queries, k={K}; eager vs mmap must "
              f"return identical ids):")
        print("   L | eager ms | mmap ms | surcharge | recall@10 | ids identical")
        for L in (32, 64, 128):
            out = {}
            for mode in ("eager", "mmap"):
                r = dap.IndexReader(path, mode=mode)
                r.beam_search(Q[0], k=K, L=L)              # warmup
                pred = np.zeros((NQ, K), dtype=np.int64)
                t0 = _now()
                for i in range(NQ):
                    ids, _ = r.beam_search(Q[i], k=K, L=L)
                    pred[i, :len(ids)] = ids
                out[mode] = ((_now() - t0) * 1000 / NQ, pred)
                r.close()
            (ms_e, pred_e), (ms_m, pred_m) = out["eager"], out["mmap"]
            same = bool(np.array_equal(pred_e, pred_m))
            rec = dap.recall_at_k(pred_e, truth, K)
            print(f" {L:>3} | {ms_e:8.2f} | {ms_m:7.2f} | {ms_m / ms_e:8.2f}x "
                  f"| {rec:9.3f} | {same}")
            assert same, "eager and mmap disagreed - storage bug"

    print("\nNotes: mmap timings are warm-cache (see results.md section 2.1);")
    print("absolute latency is Python-overhead-bound - read ratios, not ms.")
    print("Full experiments: run_experiments.py; recorded analysis: results.md")


if __name__ == "__main__":
    main()
