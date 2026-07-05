# Multi-SeRF Prototype: Multi-Attribute Range-Filtered ANN

This is a standalone Python prototype for **Multi-SeRF**, a Compound Segment
approach for vector similarity search with multiple range predicates.

The project answers one focused question:

> If a vector search query has range predicates on both `A` and `B`, can we
> reduce wasted post-filtering by partitioning the data on `B` and only
> searching buckets whose `B` range overlaps the query?

The answer from this prototype is: **yes, when the `B` predicate is selective**.
On synthetic data, Multi-SeRF reaches the recall floor while improving QPS over
the `SeRF+ResidualB` baseline by 15–25x at 1% `B` selectivity and 3.5–4x at 5%
(range over three data seeds), and the advantage grows in a single larger run
at n=20k.

## Why This Matters

Approximate nearest-neighbor (ANN) indexes are good at vector similarity, but
database workloads usually combine vector search with structured filters. A
simple post-filter strategy can waste work: the ANN search may visit many
vectors that later fail the filter.

This prototype focuses on the case where a query has two ordered range
predicates:

```text
nearest vectors to q
where A between a_lo and a_hi
  and B between b_lo and b_hi
```

The baseline uses a single SeRF-style graph for `A` and applies `B` as a
residual filter after graph traversal. Multi-SeRF instead partitions by `B`, so
it can skip unrelated buckets before graph search.

## Approach

![Schematic: baseline residual-B filtering searches the whole space, Multi-SeRF routes to the buckets overlapping the query B window](figures/fig_mechanism.png)

Multi-SeRF uses a **Compound Segment** layout:

1. Sort data by secondary attribute `B`.
2. Partition the data into `K` equal-frequency `B` buckets.
3. Build one simplified SeRF-style `SegmentGraph1D` over attribute `A` inside
   each bucket.
4. For a query, search only buckets whose `B` range overlaps `[b_lo, b_hi]`.
5. Merge candidates and rerank by exact vector distance.

This is **not** a full SeRF reimplementation. The per-bucket graph is a
deliberately simplified, SeRF-inspired `SegmentGraph1D`. The important
experimental control is that both Multi-SeRF and the `SeRF+ResidualB` baseline
share the same graph implementation, so their QPS/recall difference isolates
the value of `B`-bucket routing.

## Headline Results

Main run: `n=5000`, `dim=32`, `nq=100`, `K=16`, recall floor `0.9`, synthetic
Gaussian vectors with independent uniform attributes.

| B selectivity | SeRF+ResidualB QPS | Multi-SeRF QPS | Ratio |
|---:|---:|---:|---:|
| 1% | 25.4 | 623.6 | 24.6x |
| 5% | 142.8 | 506.4 | 3.55x |
| 10% | 154.3 | 323.3 | 2.10x |
| 25% | 300.8 | 196.1 | 0.65x |
| 50% | 281.8 | 119.8 | 0.43x |

![Headline K=16 run: QPS ratio of Multi-SeRF over SeRF+ResidualB by B selectivity, with the ratio=1 parity line and min–max whiskers over 3 data seeds](figures/fig_main_K16.png)

"QPS at recall ≥ 0.9" means each method raises its over-fetch multiplier `α`
until mean recall clears 0.9, and the QPS at that smallest `α` is reported.
The narrow-B gap is exactly this: the baseline needs `α=128–256` to survive
residual-B filtering, Multi-SeRF needs `α=8`:

![Recall vs QPS as over-fetch alpha grows, s_B=1%: the baseline reaches the 0.9 floor only at alpha=256, Multi-SeRF at alpha=8](figures/fig_recall_qps.png)

Interpretation:

- Narrow `B` predicates are the winning case. Multi-SeRF skips most buckets and
  avoids large residual-filter waste.
- Wide `B` predicates expose the cost of running multiple graph searches. When
  many buckets overlap, the single baseline graph can be faster.
- The bucket count `K` is the main design trade-off: larger `K` improves narrow
  `B` queries but hurts wide `B` queries.

The K trade-off across the K=4 / K=16 / K=32 runs:

![QPS ratio vs B selectivity for K=4, K=16, K=32: larger K wins on narrow B, smaller K holds up on wide B](figures/fig_ratio_by_K.png)

The crossover point (ratio = 1) moves with `K`: roughly 40% B selectivity for
K=4 versus roughly 13% for K=16 and K=32. No single `K` wins everywhere; the
right `K` depends on the expected `B` selectivity of the workload.

### Robustness checks

| check | result |
|---|---|
| 3 data seeds (main config) | 1% ratio spans 15.1–24.6x; 5% is a stable 3.5–4.0x. The 1% spread is grid-quantisation: the baseline needs `α=128` or `α=256` depending on seed, and the α grid doubles per step. The ≥2x criterion at `s_B ≤ 5%` holds on every seed. |
| B correlated with vectors (`--b-corr 0.8`, rank corr ≈ 0.78) | 14.5x / 3.8x / 2.8x / 0.76x / 0.41x — inside the seed-variance envelope; no degradation observed at this correlation strength. |
| n = 20,000 (single run, nq=50) | Advantage grows across the board: 15.5x / 11.2x / 6.5x / 2.9x / 0.85x. The crossover moves past 25% B selectivity, and the wide-B penalty shrinks to near parity. |

See `results.md` §9 for the full tables and the honest caveats on each check.

See `results.md` for the full write-up, caveats, and K-sensitivity tables.

## Files

| file | what |
|---|---|
| `design_notes.md` | architecture, simplified segment graph, isolation argument, scope and limits |
| `multiserf_proto.py` | `SegmentGraph1D`, `CompoundSegment`, ground truth, and recall helpers |
| `run_experiments_partB.py` | experiment runner for B-selectivity sweeps and QPS-at-recall measurement |
| `test_sanity.py` | sanity tests: `recall_at_k` semantics; K=1 equals the baseline path; predicate safety |
| `make_figures.py` | regenerates `figures/*.png` from the existing result JSON files |
| `figures/` | result figures used in this README |
| `results.md` | full result analysis, caveats, and success-criteria discussion |
| `requirements.txt` | numpy (runtime); matplotlib / pytest (optional tooling) |
| `results_partB_main.json` | headline run: K=16, full A range |
| `results_partB_K4.json`, `results_partB_K32.json` | bucket-count sensitivity runs |
| `results_partB_arestrict.json` | restrictive A-range run |
| `results_partB_seed1.json`, `results_partB_seed2.json` | data-seed variance reruns of the main config |
| `results_partB_bcorr.json` | correlated-B run (`--b-corr 0.8`) |
| `results_partB_n20k.json` | n=20,000 scale check (nq=50) |
| `results_partB_smoke.json` | small smoke run |
| `` | captured stdout for recorded experiments |

## How to Run

Requires `numpy`. No other runtime dependency is required; `SegmentGraph1D` is
hand-rolled.

```bash
# headline: B-selectivity sweep, K=16 buckets, full A range
py -3 run_experiments_partB.py --n 5000 --dim 32 --nq 100 --K 16 --out results_partB_main.json

# bucket-count sensitivity
py -3 run_experiments_partB.py --n 5000 --dim 32 --nq 100 --K 4  --out results_partB_K4.json
py -3 run_experiments_partB.py --n 5000 --dim 32 --nq 100 --K 32 --out results_partB_K32.json

# restrictive A range: A predicate selectivity 25%
py -3 run_experiments_partB.py --n 5000 --dim 32 --nq 100 --K 16 --a-sel 0.25 --out results_partB_arestrict.json

# robustness: other data seeds, correlated B, larger n
py -3 run_experiments_partB.py --n 5000 --dim 32 --nq 100 --K 16 --seed 1 --out results_partB_seed1.json
py -3 run_experiments_partB.py --n 5000 --dim 32 --nq 100 --K 16 --b-corr 0.8 --out results_partB_bcorr.json
py -3 run_experiments_partB.py --n 20000 --dim 32 --nq 50 --K 16 --out results_partB_n20k.json

# quick smoke
py -3 run_experiments_partB.py --n 1000 --dim 16 --nq 20 --K 8 --out results_partB_smoke.json
```

Key flags:

| flag | meaning |
|---|---|
| `--K` | bucket count for Multi-SeRF; K=1 is the SeRF+ResidualB baseline |
| `--a-sel` | A-range selectivity; 1.0 means full A range |
| `--b-sweep` | B-selectivity grid |
| `--b-corr` | correlation of B with the vectors; 0.0 = independent (default) |
| `--seed` | data seed (query seed is independent and fixed) |
| `--n`, `--dim`, `--nq`, `--k` | synthetic dataset and query settings |

### Tests and figures

```bash
# sanity tests (fast; needs pytest)
py -3 -m pytest test_sanity.py -q

# regenerate figures/*.png from the recorded results_partB_*.json (needs matplotlib)
py -3 make_figures.py
```

The tests pin what the experiment's validity rests on: `recall_at_k` measures
against the achievable truth set, `CompoundSegment` K=1 reduces exactly to the
SeRF+ResidualB baseline path, and no query result ever violates the range
predicates. The figure script only reads the recorded JSON files; it never
reruns experiments.

## What Is Supported

- The main success criterion is met: Multi-SeRF is at least 2x faster than
  SeRF+ResidualB for narrow `B` ranges (`s_B <= 5%`) while reaching recall 0.9
  — and this holds on all three data seeds tested and with B correlated to the
  vectors.
- The bucket-routing mechanism is isolated because Multi-SeRF and the baseline
  use the same `SegmentGraph1D` code.
- The K trade-off is quantified with K=4, K=16, and K=32 runs.
- A restrictive A-range run exercises both A filtering and B bucket routing.
- A single n=20k run shows the advantage growing with scale, in the direction
  the design predicts.

## Limitations

- This is a single-thread Python research prototype, not a production vector
  database.
- The data is synthetic. Attribute–vector correlation was tested at one
  pattern and strength (Gaussian copula, ρ=0.8 on one coordinate); real
  datasets remain untested.
- Only the main configuration has multi-seed variance data (3 seeds); the K
  sweeps, A-restrict, and n=20k runs are single runs.
- `SegmentGraph1D` is SeRF-inspired but not a faithful SeRF-2D implementation.
- Absolute QPS should not be compared to production ANN systems. The meaningful
  evidence is the like-for-like ratio between Multi-SeRF and SeRF+ResidualB.
- Wide `B` ranges are not a win for K=16 or K=32 at n=5k; the n=20k run
  narrows this penalty to 0.85x but is a single run, still far from the
  n=1M+ regime where SeRF-class methods are usually evaluated.

## Portfolio Summary

One-sentence version:

> Built a Python prototype for multi-attribute range-filtered vector search,
> showing that `B`-bucket routing improves QPS by 15–25x over a residual-filter
> baseline at 1% secondary-predicate selectivity (3 seeds, recall ≥ 0.9), with
> the advantage growing at larger n.

Best way to present the project:

- Lead with the multi-filter vector search problem.
- Explain the `B`-bucket routing idea visually or with a small diagram.
- Show the QPS ratio table and the K trade-off.
- Be explicit that this is a prototype with synthetic data and shared-code
  baselines, not a complete DBMS integration.
