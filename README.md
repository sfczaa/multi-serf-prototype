# Vector Search Index Prototypes

This repository contains two research prototypes for vector search inside
database systems. For portfolio use, the recommended focus is **Part B:
Multi-SeRF**, because it has the clearest problem statement, baseline, and
experimental result.

## Portfolio Focus: Part B Multi-SeRF

**Problem.** Vector similarity search often needs structured filters, for
example "find nearest vectors where attribute A and attribute B are both in a
range." A common approach is to search a vector index first, then post-filter
by the structured predicates. This wastes work when the second range predicate
is selective.

**Idea.** Multi-SeRF uses a Compound Segment layout:

1. Partition the dataset by a secondary ordered attribute `B`.
2. Build a simplified SeRF-style segment graph over primary attribute `A`
   inside each bucket.
3. At query time, only search buckets whose `B` range overlaps the query.

This isolates the benefit of `B`-bucket routing against a `SeRF+ResidualB`
baseline that uses the same per-bucket graph code but applies `B` only as a
residual filter.

![Schematic: baseline residual-B filtering vs Multi-SeRF B-bucket routing](PartB_MultiSeRF/figures/fig_mechanism.png)

**Headline result.** On synthetic data, Multi-SeRF is much faster when the
`B` predicate is narrow (recorded headline run; whiskers below show the range
over three data seeds):

| B selectivity | Multi-SeRF vs SeRF+ResidualB |
|---:|---:|
| 1% | 24.6x QPS (15–25x over 3 seeds) |
| 5% | 3.55x QPS (3.5–4.0x) |
| 10% | 2.10x QPS (2.1–2.7x) |
| 25% | 0.65x QPS (0.65–0.76x) |
| 50% | 0.43x QPS (0.38–0.43x) |

![Headline K=16 run: QPS ratio of Multi-SeRF over SeRF+ResidualB by B selectivity, with min–max whiskers over 3 data seeds](PartB_MultiSeRF/figures/fig_main_K16.png)

The result supports the main claim: bucket routing helps most when the second
range predicate is selective. It also shows the trade-off honestly: when the
`B` predicate is wide, Multi-SeRF searches many buckets and can become slower
than a single baseline graph — and then **resolves that trade-off** with a
query-adaptive router that keeps both indexes and picks per query, reaching
27x on narrow windows while staying at parity (1.00x) on wide ones. Robustness
checks (three data seeds, a correlated-B run, real SIFT10K vectors, larger-n
runs, tau sensitivity, and mixed workloads) are in
[PartB_MultiSeRF/results.md](PartB_MultiSeRF/results.md).

See [PartB_MultiSeRF/README.md](PartB_MultiSeRF/README.md) for the full Part B
orientation and [PartB_MultiSeRF/results.md](PartB_MultiSeRF/results.md) for
the experiment write-up. Figures are regenerated from the recorded result JSON
files by `PartB_MultiSeRF/make_figures.py`, and `PartB_MultiSeRF/test_sanity.py`
holds the sanity tests.

## Secondary Prototype: Part A DiskANN-Style On-Disk ANN

Part A explores an on-disk ANN index layout inspired by DiskANN. It implements
the algorithmic kernel in Python: Vamana graph construction, product
quantization, page-aligned file layout, mmap-backed reads, beam search, and
full-vector reranking.

This is best presented as an advanced systems prototype, not as a finished
DuckDB extension. The storage layout round-trips correctly (byte-exact, now
pinned by sanity tests), the file-size claim is supported, and on real
SIFT10K vectors the proposal's recall bar clears decisively (0.998 at L=64;
the miss reported on synthetic Gaussian was a property of the data). DuckDB
integration, cold-cache benchmarking, MVCC, and beyond-10k-scale validation
remain out of scope. It ships with a one-minute `demo.py`, storage-layer
sanity tests, and figures generated from the recorded runs.

See [PartA_DiskANN/README.md](PartA_DiskANN/README.md).

## Repository Layout

| path | purpose |
|---|---|
| `PartB_MultiSeRF/` | Primary portfolio project: multi-attribute range-filtered ANN prototype |
| `PartA_DiskANN/` | Secondary prototype: on-disk ANN algorithmic kernel |
| `.github/workflows/ci.yml` | CI: Part A/B sanity tests + Part B smoke demos + figure regeneration |

## Recommended Portfolio Positioning

Use Part B as the main story:

> I built a Python prototype of a multi-attribute range-filtered vector search
> index. It partitions data by a secondary range attribute and searches only
> relevant buckets, improving QPS by 15–25x over a residual-filter baseline
> when the secondary predicate is selective — reproduced on real SIFT vectors,
> growing with n (at n=100k it is faster wherever both arms reach recall 0.9,
> and at 1–5% it is the only arm that does), topped with a query-adaptive
> router that removes the wide-window penalty.

Keep the scope precise:

- This is a research prototype, not a production vector database.
- Results are from single-thread Python on synthetic data.
- The key evidence is the like-for-like ratio between Multi-SeRF and
  SeRF+ResidualB, which share the same graph implementation.
- Wide-range queries expose a real trade-off: more buckets searched means more
  overhead.
