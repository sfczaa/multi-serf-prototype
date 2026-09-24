# Part A Prototype: Design Notes

> Standalone (not inside DuckDB) DiskANN-style on-disk ANN index.
> The goal is to make the Part A proposal's three load-bearing claims testable on a single machine without touching DuckDB's C++ source tree:
>   1. on-disk graph + PQ + full-precision rerank can keep peak resident memory bounded;
>   2. Beam Search with PQ pruning + full-precision rerank retains recall ≥ 0.95;
>   3. query latency stays within 3-5× of an in-memory graph baseline at the same recall.
>
> What is intentionally *not* in scope here: DuckDB block manager integration, MVCC, transactions, multi-shard merge, and a SQL planner hook. Those are the items in the proposal that genuinely require modifying DuckDB and are out of scope for this prototype.

---

## 1. Why a standalone prototype

The full Part A vision is a `vss` extension fork that plugs PQ-compressed graph traversal into DuckDB's block manager. That work is dominated by two costs:

- C++ extension boilerplate: `CREATE INDEX` parsing, catalog entries, snapshotting, MVCC versioning, planner rewrite for `ORDER BY array_distance(...) LIMIT k`.
- Block manager surgery: making the on-disk graph live in DuckDB's `.duckdb` page file and route through the shared buffer manager.

The prototype tests the graph-search algorithm with a flat file. DuckDB integration through `BufferManager::Pin(block_id)` is unimplemented; a 1-to-1 mapping from prototype reads to buffer-manager calls has not been validated.

## 2. On-disk layout

One file, prototype page size `P = 4096 B`. All segments are page-aligned. Multi-byte integers are little-endian. DuckDB's [default storage block size](https://duckdb.org/docs/lts/configuration/pragmas#block-sizes) is 256 kB, so this layout does not establish equivalent DuckDB I/O behavior.

```
+---------------------------------------------+  page 0
| HEADER (96 B, rest of page zero-padded)     |
|   magic        u32   = 0x44414E50 ("DANP")  |
|   version      u32   = 1                    |
|   n            u64   = #vectors             |
|   dim          u32                          |
|   M            u32   degree cap             |
|   ep           u32   entry-point node id    |
|   pq_m         u32   PQ subquantizers       |
|   pq_ks        u32   codes per subq (=256)  |
|   graph_off    u64   byte offset            |
|   pq_off       u64                          |
|   full_off     u64                          |
|   codebook_off u64                          |
+---------------------------------------------+
| GRAPH segment                               |
|   one fixed-size record per node:           |
|     deg     u16                             |
|     pad     u16                             |
|     nbrs[M] u32                             |
|   record size = 4 + 4*M bytes, padded to    |
|   a multiple of cache-line (64 B).          |
|   nodes are packed page-aligned: nodes per  |
|   page = floor(P / rec_size); unused tail   |
|   bytes per page are zero.                  |
+---------------------------------------------+
| PQ segment                                  |
|   row-major u8[n, pq_m]                     |
|   packed contiguously, page boundaries do   |
|   not split a node's code (we round n_per_  |
|   page down so each row is intact).         |
+---------------------------------------------+
| FULL segment                                |
|   row-major float32[n, dim]                 |
|   page-aligned, same packing rule as PQ.    |
+---------------------------------------------+
| CODEBOOK segment                            |
|   float32[pq_m, pq_ks, dim/pq_m]            |
|   small (a few hundred KB), pinned in RAM   |
|   after open.                               |
+---------------------------------------------+
```

Why three segments instead of one fat node record. Beam Search walks the *graph* segment hot, scores candidates against the *PQ* segment hot, and only touches *full* vectors for the top-L rerank. Co-locating those three streams in one record would force every neighbor-list read to also drag the full 4 KB-ish vector into cache, which is exactly the regression we are trying to avoid. Separating them lets the prototype's page cache behave the way DuckDB's buffer pool would behave with three different access patterns.

Versioning. The `version` field in the header is what the Part A proposal's "Storage versioning" challenge cashes out as in practice - a future change to the record format bumps it and the loader refuses to open the file. Adding it now costs nothing.

## 3. Build pipeline

```
BuildIndex(X[n,dim], M, ef_construction, pq_m):
    train PQ codebook on a sample (~50k rows) of X
    encode all X into PQ codes
    initialize G with n nodes, each seeded with M random neighbors
    pick entry point ep = node nearest to centroid of X
    VamanaPass(alpha=1.0)
    VamanaPass(alpha=1.2)
    write file: header → graph → PQ → full vectors → codebook

VamanaPass(alpha):
    for each point p in a fresh random order:
        V = GreedySearch(G, p, ef_construction, start=ep)
        V = V ∪ G[p].nbrs                  # keep what the earlier pass found
        set G[p].nbrs = RobustPrune(p, V, alpha, M)
        for each q in G[p].nbrs:
            G[q].nbrs.add(p)
            if |G[q].nbrs| > M: G[q].nbrs = RobustPrune(q, G[q].nbrs, alpha, M)
```

This is the in-memory variant of the Vamana build, run twice: once at α=1.0 and again at α=1.2, which is the DiskANN paper's Section 3 procedure. The second pass keeps each node's current neighbors in the candidate set, so it tightens the graph the first pass built instead of rebuilding it. The two passes carry real weight: on the older in-set runs, a single α=1.2 pass measured recall@10 = 0.695 at n=10k against 0.900 for the pair (`results.md` §2.3). The proposal's "chunked, shard-then-merge" build is what you'd use when even the graph doesn't fit in RAM - we don't need it at the scale this prototype runs (≤ 10k vectors), and the chunked builder is not implemented. It is listed under §6 "What this prototype does NOT prove" as future work; the algorithmic reference is DiskANN §3.

RobustPrune is the standard Vamana α-pruning rule: keep a neighbor only if no already-kept neighbor is α-closer to it than the source node is. α=1.2 is the DiskANN default.

## 4. Beam Search query

```
Query(q, k, L):                              # L = beam width, L ≥ k
    open file, mmap-mode for graph + PQ,
    lazy reads for full segment
    asymmetric_table = precompute_asym_table(q, codebook)
                                              # pq_m * pq_ks float32
    visited = bitset(n)
    pool = max-heap of (pq_distance, node) keyed by distance, size ≤ L
    pool.push((asym(ep), ep))
    visited[ep] = true
    while ∃ unexpanded node in pool:
        u = pool.pop_closest_unexpanded()
        nbrs = read_graph_page(u)             # one page read
        for v in nbrs:
            if visited[v]: continue
            visited[v] = true
            d = asym_pq_distance(v, asym_table)   # u8 lookups → float sum
            if pool.size < L or d < pool.worst():
                pool.push((d, v))
                if pool.size > L: pool.pop_worst()
    # rerank
    top_L = pool.sorted_ascending()[:L]
    for v in top_L:
        v.dist = exact_l2(q, read_full_vector(v))
    return top_k_by(top_L, exact_l2)[:k]
```

Where the disk reads live. `read_graph_page(u)` is the only frequently-paged call; the PQ codes for *any* candidate ever pushed into the pool are read from the mmap'd PQ region, but those reads stream sequentially through the OS page cache. `read_full_vector(v)` happens exactly `L` times per query, near the end - this is what the proposal calls "Step 3: rerank using full vector for top-L candidates".

Asymmetric PQ distance. Standard PQ trick: precompute a `pq_m × pq_ks` table of squared distances between the query subvectors and the codebook entries, so per-candidate scoring is `pq_m` table lookups + a sum. This is the part that gives "PQ distance in-cache" the meaning it has in the proposal - the codebook is small (≈ KB), pinned, and the lookups are L1-resident.

## 5. Experiment plan

| | scope of prototype | scope of full Part A |
|---|---|---|
| dataset size | 1k-10k vectors (as run) | 1M-10M (proposal target); 50M aspirational |
| dim | 128 (SIFT) | same |
| storage | OS file (mmap) | DuckDB block manager |
| build mode | in-memory Vamana | chunked + shard merge |
| query path | Python + numpy | DuckDB operator + planner hook |
| concurrency | single-thread | MVCC-aware |

### Datasets

1. Random Gaussian, n × 128. The runner generates a fresh standard-normal dataset (default n = 10k) plus a held-out Gaussian query set (independent seed) so queries are not present in the index. This is the default and carries every number in `results.md` §1-§5.
2. SIFT / GIST. The proposal references SIFT1M / GIST1M as the natural benchmark. A SIFT-style `.fvecs` loader (`load_fvecs` / `load_sift`) is in `run_experiments.py`, along with `--dataset sift` and `--sift-dir` CLI flags. One real-data run is recorded: TEXMEX `siftsmall`, 10k base vectors and the corpus's own 100 queries (`results.md` §8). With the same hyperparameters, recall@10 is 0.998 on SIFT10K and remains below 0.95 on the Gaussian runs. SIFT1M and GIST stay untested.

### Baselines

| name | what it tests |
|---|---|
| `exact` | numpy brute force - ground truth and lower-bound recall = 1.0 |
| `pynndescent` | in-memory graph-based ANN, stands in for "in-memory HNSW" since hnswlib doesn't build on this machine |
| `proto-eager` | our index, all segments loaded into numpy arrays at open time - pure in-memory baseline that isolates algorithmic cost from I/O cost |
| `proto-mmap`  | our index, segments mmap'd; each graph/PQ/full read may trigger a page fault and go through the OS page cache |

Originally the design called for a `proto-cold` baseline that explicitly dropped the OS page cache before each query to model the worst-case disk-bound path. That was not implemented - `proto-mmap` queries reuse whatever the OS page cache happens to hold, so its numbers sit somewhere between warm and cold and should not be read as a cold-cache measurement. A real cold-cache test would need either `posix_fadvise(DONTNEED)` plus a cache drop between queries, or a dataset large enough that the working set exceeds RAM. Neither is in scope for the prototype.

### Metrics

- recall@10 (vs exact ground truth), averaged over query set
- per-query latency: mean and p99 (ms)
- build time (s)
- on-disk footprint (MB), broken down by segment
- peak RSS during query (via `psutil` if available, else `tracemalloc` upper bound)

### Success conditions (prototype-level)

These are targets, not guarantees; see `results.md` for what the prototype actually hit:

- recall@10 reasonably close to the proposal's 0.95 target on the `proto-eager` / `proto-mmap` paths; we accept that Gaussian + small n may force a larger `L` to clear the bar than SIFT-scale would
- `proto-eager` latency in the same order of magnitude as `pynndescent` (pure Python overhead means exact ratios are not meaningful)
- `proto-mmap` latency within a small multiple of `proto-eager`, quantifying the mmap surcharge, *not* a cold-cache disk surcharge
- on-disk footprint approximately `n × (4·M + pq_m + 4·dim) + overhead`; we verify the constant factor

`proto-eager` and `proto-mmap` are the same algorithm over the same bytes, so any recall difference would be a bug; latency differences are attributable to the storage path. This setup validates the *algorithmic* half of Part A and gives a lower-bound read on the I/O cost. The *systems* half - DuckDB buffer pool, MVCC, cold-cache behaviour at scale - remains future work.

## 6. What this prototype does NOT prove

Calling these out explicitly so the results writeup doesn't overclaim:

- It doesn't prove the index can live inside a `.duckdb` file: we use a sibling OS file.
- It doesn't prove MVCC compatibility: there are no concurrent writers.
- It doesn't prove shard-merge correctness: the chunked / shard-then-merge builder is not implemented, not even as a stub.
- It doesn't reach benchmark scale on real data. The one real-data run is `siftsmall`: 10k vectors, 100 corpus queries, single run, warm cache (`results.md` §8). SIFT1M and GIST are untested, and every number in §1-§5 of `results.md` is on synthetic Gaussian.
- It doesn't measure cold-cache disk-bound behaviour: `proto-mmap` runs against whatever the OS page cache happens to hold; there is no `proto-cold` baseline that flushes the cache.
- It doesn't prove the storage format is forward-compatible across vss releases: only the `version` field is in place.
- It doesn't scale-test to 1M-10M; that requires moving the build off Python.

These map directly onto the "Expected Challenges" section of the Part A proposal and are the right items to list as the implementation roadmap that this prototype seeds.
