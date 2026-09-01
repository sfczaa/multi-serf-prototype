# Part B Prototype — Multi-SeRF Design Notes

> Standalone (not inside any DBMS) prototype of **Multi-SeRF**, the Compound
> Segment (CS) structure proposed in the accompanying Part B proposal.
> The goal is to make the proposal's one load-bearing claim testable on a
> single machine:
>
> > A segment-graph index can treat **two** ordered range attributes as
> > first-class index dimensions — partition on the secondary attribute *B*,
> > build a SeRF-style segment graph over the primary attribute *A* inside
> > each bucket — and beat the single-attribute SeRF baseline (which can only
> > index *A* and must apply *B* as a residual filter), **most strongly when
> > the *B* predicate is selective**.
>
> What is intentionally *not* in scope: a faithful re-implementation of SeRF's
> full 2D segment graph (MaxLeap/MidLeap/MinLeap), the real Amazon/Airbnb
> datasets named in the proposal, DBMS/SQL integration, persistence, and
> dynamic updates.

---

## 1. What we are actually trying to measure

The novel object in Part B is the **Compound Segment**. Its claim is *not*
"segment graphs are good at range-filtered ANN" — that is SeRF's claim, already
established in SIGMOD 2024. Multi-SeRF's claim is narrower and specific:

> Given a query with range predicates on **both** *A* and *B*, routing by
> *B*-bucket containment and only searching the buckets that overlap
> `[b_lo, b_hi]` is faster than searching one big *A*-index and filtering *B*
> out afterwards — and the gap widens as the *B* range gets narrower.

So the experiment must isolate **the contribution of B-bucketing**, holding
everything else fixed.

### 1.1 The isolation argument (the crux of this prototype)

This mirrors the cleanest comparison from the Part A prototype (there it was
`proto-eager` vs `proto-mmap`: same algorithm, different storage; the delta was
pure I/O cost). Here:

> **Every graph-based method in this prototype uses the *same*
> `SegmentGraph1D` implementation for the A-attribute index.** The methods
> differ *only* in how the two predicates are applied:

| method | how A is handled | how B is handled |
|---|---|---|
| HNSW + PostFilter | not used (full-range query = plain ANN) | residual filter, *after* the walk |
| SeRF + ResidualB  | segment graph (the A range) | residual filter, *after* the walk |
| **Multi-SeRF (CS)** | segment graph (the A range), per bucket | **bucket routing** + boundary residual |

Because `SegmentGraph1D` is byte-for-byte the same code in all three, any
QPS/recall difference between **Multi-SeRF** and **SeRF+ResidualB** is
attributable to the B-bucketing alone. This means the prototype's headline
claim does **not** depend on `SegmentGraph1D` being a perfect reproduction of
SeRF — it only has to be a *consistent, reasonable* range-filtered ANN that is
used identically across arms. That is a much weaker (and honestly defensible)
requirement than "we re-implemented SeRF correctly."

---

## 2. SegmentGraph1D — the simplified segment graph

### 2.1 What full SeRF does, and what we simplify

SeRF's segment graph encodes *n* HNSW graphs — one per left-bounded range —
as a single graph in which **each edge carries a validity interval** over the
attribute. Its 2D segment graph generalises this to arbitrary `[lo, hi]`
ranges in average-case `O(n log n)` space using the MaxLeap/MidLeap/MinLeap
construction. That 2D construction is the genuinely intricate part of the
paper and the part we deliberately do **not** reproduce.

Our `SegmentGraph1D` keeps the **core idea that makes SeRF work** — *one graph,
edges tagged with validity intervals, navigable for any sub-range* — but
simplifies in three documented ways:

1. **Single layer, not hierarchical.** We build a flat NSW-style proximity
   graph, not a multi-layer HNSW. The hierarchy is a large-scale latency
   optimisation; at the prototype's per-bucket scale (hundreds–thousands of
   points) a single greedy layer with a reasonable `ef` recovers high recall.
2. **Lower bound is exact, upper bound is residual.** The segment graph
   natively handles the **lower** bound `a_lo` (nodes with `a < a_lo` and edges
   that have expired simply do not exist in the queried graph). The **upper**
   bound `a_hi` is applied as a cheap *accept-test* during the walk: a node is
   added to the result pool only if `a ≤ a_hi`. Full SeRF's 2D graph handles
   both bounds natively; we handle one natively and one by residual filtering.
   *This asymmetry is shared by every arm that uses `SegmentGraph1D`, so it
   cancels out of the Multi-SeRF-vs-baseline comparison.*
3. **No deletion / append-only.** Matches SeRF's own L2 limitation; out of
   scope here.

### 2.2 Construction — edge validity intervals

Insert points in **descending `a` order**: `p_(1)` has the largest `a`, …,
`p_(m)` the smallest. After inserting the first `t` points, the graph holds
exactly the points with the `t` largest `a` values — i.e. it *is* the graph for
the left-bounded query `a_lo = a_(t)`.

Each **directed edge** `(u → v)` records an interval `[death_a, birth_a]`:

- `birth_a` = the `a`-value of the point being inserted when the edge was
  created (the just-inserted point always has the smallest `a` so far, so it is
  the binding endpoint).
- `death_a` = the `a`-value of the point whose insertion pruned this edge
  (when `u`'s out-degree overflowed `M` and `v` was evicted). If the edge is
  never pruned, `death_a = -∞`.

At query threshold `a_lo`, edge `(u → v)` is **active** iff
`death_a < a_lo ≤ birth_a`, and node `v` exists iff `a_v ≥ a_lo`. Intuition:
the edge appears once we have inserted the point that created it (threshold has
dropped to `birth_a`) and disappears once we insert the point that pruned it
(threshold reaches `death_a`).

This is the faithful part: it is exactly SeRF's "one graph, edges valid over a
contiguous attribute interval" mechanism, restricted to the left-bounded case.

### 2.3 Query

```
SegmentGraph1D.query(q, a_lo, a_hi, k, ef):
    entry = the max-a node            # always exists for any a_lo ≤ a_max
    greedy beam search from entry, beam width ef:
        when expanding u, follow only edges active at a_lo
        only traverse into nodes v with a_v ≥ a_lo
        push v into the result pool only if a_v ≤ a_hi   # upper-bound residual
    return the k pool members closest to q (exact L2 on the full vector)
```

`ef ≥ k`; we over-fetch with `ef = max(ef_floor, ⌈α·k⌉)` so the upper-bound
residual does not starve the pool.

---

## 3. Compound Segment

```
BuildCS(D = {(v_i, a_i, b_i)}, K, M, ef):
    sort D by b ascending
    partition into K contiguous equal-frequency buckets B_1..B_K  (each ⌊n/K⌋ or ⌈n/K⌉)
    for each bucket B_j:
        SG_j   = SegmentGraph1D.build(vectors in B_j, attribute a, M, ef)
        r_j    = [min b in B_j, max b in B_j]            # inclusive B-range
    return ({B_j}, {SG_j}, {r_j})
```

```
QueryCS(q, [a_lo,a_hi], [b_lo,b_hi], k, α):
    k' = ⌈α·k⌉                                            # per-bucket over-fetch
    cand = ∅
    for each bucket j:
        if r_j ∩ [b_lo,b_hi] = ∅:   continue              # skip — the whole point
        C_j = SG_j.query(q, a_lo, a_hi, k')
        if r_j ⊄ [b_lo,b_hi]:                             # boundary bucket
            C_j = { x in C_j : b(x) ∈ [b_lo,b_hi] }       # residual filter on B
        cand ∪= C_j
    return top-k of cand by exact L2 to q
```

- **Interior** bucket (`r_j ⊆ [b_lo,b_hi]`): every point already satisfies the
  B predicate, so no residual filter is needed.
- **Boundary** bucket (overlaps but not contained): residual-filter on B.
- **Disjoint** bucket: skipped entirely — this is where the QPS win comes from
  when `[b_lo,b_hi]` is narrow.

`α ∈ [1.5, 3]` is the only Multi-SeRF parameter that does not appear in pure
SeRF; it compensates for candidates lost to the boundary residual filter.

---

## 4. Baselines

All four share the same dataset and the same ExactScan ground truth. The three
graph methods share `SegmentGraph1D` (see §1.1).

| name | what it tests |
|---|---|
| `ExactScan` | brute-force L2 over all points, then filter both predicates. Ground truth; recall = 1.0 by definition. O(n) every query. |
| `RangeFirstScan` | apply both range predicates first, then exact L2 on the survivors. Also exact (recall 1.0); its *latency* is the "range-first" strategy — cheap at low selectivity, expensive at high. |
| `HNSW+PostFilter` | `SegmentGraph1D` queried with the **full** A-range (= plain ANN), then post-filter both A and B. Recall degrades when selectivity is low (few of the ANN top results survive the filter). |
| `SeRF+ResidualB` | **one** `SegmentGraph1D` over *all* points on attribute A; query `[a_lo,a_hi]`, then residual-filter B. This is SeRF's L1 limitation made concrete and is the **head-to-head baseline** for Multi-SeRF. |

---

## 5. Experiment plan

| | scope of prototype | scope of full Multi-SeRF |
|---|---|---|
| dataset | synthetic Gaussian vectors + 2 synthetic ordered attrs | Amazon ~500K, Airbnb ~200K (real embeddings) |
| n | 10k–50k | 200k–500k |
| dim | 64–128 | 384 |
| index | in-memory `SegmentGraph1D` | persisted SeRF 2D segment graph |
| A-range handling | segment graph (lower) + residual (upper) | full 2D segment graph |

### 5.1 Dataset

Synthetic, controllable selectivity:
- vectors: `N(0, I)` in `R^d`.
- attributes `a`, `b`: independent `Uniform[0, 1]` (independent of the vectors
  and of each other). Independence is the simplest baseline; correlated
  attributes are a documented future variant.
- queries: held-out `N(0, I)` vectors (same discipline as Part A — never drawn
  from the indexed set).

### 5.2 Query workload — the B-selectivity sweep

The proposal's success criteria are stated in terms of the **B selectivity**
`s_B`. We sweep `s_B ∈ {1%, 5%, 10%, 25%, 50%}` by centring a B-range of the
required width at a random point. To isolate B-bucketing cleanly, the headline
sweep keeps the **A range full** (so SeRF+ResidualB does no A-filtering and the
only difference between it and CS is residual-B vs bucket-routing). A secondary
config uses a restrictive A range (~25%) to also exercise the segment graph's
A-filtering path.

### 5.3 Metrics

- **recall@10** vs ExactScan ground truth, averaged over the query set.
- **mean query latency (ms)** and **QPS** (= nq / total query time). The
  proposal reports "QPS at recall ≥ 0.9"; we report QPS together with the
  recall actually achieved so the reader can see both.
- **index size**: analytical bytes (edges × 4 + interval arrays), since the
  prototype is in-memory. Reported as an estimate, not a serialised on-disk
  figure.

### 5.4 Success criteria (from the proposal, restated)

The proposal calls the project successful if **either** holds:

1. Multi-SeRF achieves **≥ 2× QPS** over SeRF+ResidualB at `s_B ≤ 5%` with a
   recall floor of 0.9, **or**
2. Multi-SeRF is **within 20%** of SeRF+ResidualB QPS at `s_B ≥ 50%`.

The intuition the prototype is checking: at narrow `s_B`, CS skips almost every
bucket (big win); at wide `s_B`, CS visits almost every bucket and should merely
*match* the baseline (it must not be much worse).

---

## 6. What this prototype does NOT prove

Stated up front so the results writeup does not overclaim:

- It does **not** reproduce SeRF's 2D segment graph; the A upper-bound is a
  residual filter, not a native graph operation. Absolute numbers therefore
  understate what a faithful SeRF-2D core would achieve on the A axis — but the
  Multi-SeRF-vs-baseline *delta* is unaffected (shared code, §1.1).
- It does **not** use the real Amazon/Airbnb datasets; attributes are synthetic
  and independent of the vectors, which is the easy case for bucketing.
- No DBMS/SQL integration, persistence, or dynamic updates (the proposal's own
  L2 and the "DBMS integration" gap).
- Single-machine, single-thread, pure Python — latencies are overhead-bound and
  should be read as **ratios** (CS vs SeRF+ResidualB), not absolute QPS.
- Single run per cell unless stated; no variance/error bars.
