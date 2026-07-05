# DiskANN-style on-disk ANN — Part A prototype

Standalone Python prototype of the on-disk ANN index proposed in Part A
(`hw4_course_proposal_PartA.pdf`). Builds a two-pass Vamana graph, encodes vectors
with product quantisation, writes everything to a single page-aligned binary
file, and serves nearest-neighbour queries with PQ-pruned beam search +
full-precision rerank. The future DuckDB extension would swap the OS-file
reader for a `BufferManager::Pin(block_id)` call on the same byte layout.

This repo is the **algorithmic kernel**, not the DuckDB extension. See
`design_notes.md` for what is and is not in scope.

## Files

| file | purpose |
|---|---|
| `design_notes.md` | storage layout, build pipeline, query algorithm, experiment plan, explicit scope-limits |
| `results.md`      | experiment write-up; read §0 for caveats before the headline tables |
| `diskann_proto.py`| implementation: PQ, Vamana builder, page-aligned writer, mmap reader, beam search, ground-truth `exact_knn`, `recall_at_k` |
| `run_experiments.py` | CLI runner: dataset → build → benchmark sweep → JSON + markdown output |
| `requirements.txt` | pinned dependency floor |
| `results_*.json`, `` | raw outputs from past runs |
| `proto*.idx` | built index files (regenerable; safe to delete) |

## Install

```
python -m pip install -r requirements.txt
```

Tested on Python 3.12.7, NumPy 2.1.2, scikit-learn 1.8.0, Windows 11. No SIMD,
no JIT — Python overhead dominates absolute latency at these scales.

## Run

Synthetic Gaussian (default — fast, no external data needed):

```
python run_experiments.py --n 5000 --nq 200 --L 128 \
    --out results_5k_heldout_new.json \
    --index proto_5k_heldout_new.idx --rebuild
```

(The `_new` suffix is to avoid clobbering the existing
`results_5k_heldout.json` / `results_5k.json` files kept for reference in
this directory — pick any unused name for your own run.)

Useful flags:

| flag | meaning |
|---|---|
| `--n`, `--dim`, `--nq`, `--k` | dataset / query size; only used for `--dataset gaussian` |
| `--M`, `--ef-construction`, `--pq-m`, `--L` | Vamana / PQ / beam-search hyperparameters |
| `--rebuild` | delete and rebuild the index file before benchmarking |
| `--skip-pynnd` | skip the pynndescent in-memory baseline (saves ~30s build) |
| `--dataset {gaussian,sift}` | data source |
| `--sift-dir PATH` | directory containing `siftsmall_base.fvecs` / `siftsmall_query.fvecs` (or the full `sift_base.fvecs` / `sift_query.fvecs` pair); falls back to gaussian with a warning if neither pair is present |

Queries are drawn from an **independent** Gaussian (seed = 12345), not sampled
from the indexed set. SIFT mode uses the dataset's own query file. Sampling
queries from the index inflates recall and is the kind of mistake that gets
flagged in review — `run_experiments.py` no longer supports that mode.

### SIFT example

Download `siftsmall.tar.gz` from
<https://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz> (or the full SIFT1M), extract so that the
`.fvecs` files sit in a directory, and pass:

```
python run_experiments.py --dataset sift --sift-dir ./siftsmall \
    --L 128 --out results_siftsmall.json --index proto_sift.idx --rebuild
```

SIFT results are not yet included in `results.md` — the loader works but no
SIFT runs have been recorded.

## Reading the results

`results.md` §0 lists the caveats that apply to every number in the document.
The most load-bearing ones are: only synthetic Gaussian has been benchmarked,
the `proto-mmap` numbers are warm-cache (no cold-disk `proto-cold` mode is
implemented), and the older n=5k / n=10k tables predate the held-out-query
fix. Treat the prototype as a credibility check on the algorithmic kernel,
not a SIFT-grade benchmark.

## What's not in this repo

DuckDB block-manager integration, MVCC, chunked / shard-merge build, real
benchmark datasets beyond SIFT loading, multi-thread builds. All are
deliberately out of scope; see `design_notes.md` §6 and `results.md` §3
"Not tested".
