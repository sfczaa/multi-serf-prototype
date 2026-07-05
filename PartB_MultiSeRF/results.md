# Part B Prototype — Multi-SeRF Experiment Results

Standalone Python prototype of the **Compound Segment (CS)** structure proposed
in Part B (`hw4_course_proposal_PartB.pdf`). The CS partitions data on a secondary
ordered attribute *B* into *K* equal-frequency buckets and builds a simplified,
SeRF-inspired segment graph (`SegmentGraph1D`) over the primary attribute *A*
inside each bucket; a query searches only the buckets whose *B*-range overlaps
`[b_lo, b_hi]`.

Hardware: laptop, single-thread Python 3.12 + NumPy. The graph traversal is
pure Python; absolute QPS is overhead-bound. **Read ratios (Multi-SeRF vs
SeRF+ResidualB), not absolute QPS.**

Data: synthetic. `N(0, I)` vectors (dim 32), two **independent** `Uniform[0,1]`
ordered attributes. This is the easy case for bucketing (B is uncorrelated with
the vectors); the real Amazon/Airbnb datasets in the proposal were not used.

---

## 0. Caveats up front (read before the tables)

- **The valid comparison is Multi-SeRF vs SeRF+ResidualB.** Both are pure-Python
  and share the *same* `SegmentGraph1D` code (design_notes §1.1); their ratio
  isolates the B-bucketing. Comparisons to `ExactScan` / `RangeFirstScan` mix in
  a Python-vs-NumPy implementation gap and should not be read as absolute verdicts
  (see §6 — at this scale vectorised exact scan actually *wins* outright).
- **`SegmentGraph1D` is not SeRF.** It is a single-layer proximity graph with
  edge validity intervals that handles the A *lower* bound natively and the A
  *upper* bound as a residual filter. Faithful to SeRF's "one graph, interval-
  tagged edges" idea, simplified in the ways listed in design_notes §2.
- **Scale.** n = 5,000. Graph ANN does not pay its overhead off against brute
  force until n is far larger (SeRF's own results are at n = 1M–10M). The
  bucketing *ratio* is the transferable result; the absolute QPS is not.
- **Single run per cell**, nq = 100 held-out queries, no variance bars.
- **QPS at recall ≥ 0.9.** Each ANN method increases its over-fetch multiplier
  `alpha` until mean recall ≥ 0.9, and we report the QPS at that smallest
  (fastest) `alpha`. This is the proposal's reporting axis.

---

## 1. Headline: B-selectivity sweep (K = 16, full A range)

`results_partB_main.json` / ``. n=5000, dim=32, nq=100, k=10,
M=16, ef_build=64. `α` is the over-fetch needed to reach recall 0.9; `buckets`
is the mean number of the 16 buckets actually searched.

| s_B | pts pass | SeRF+ResidualB QPS @0.9 (α, recall) | Multi-SeRF QPS @0.9 (α, buckets, recall) | **CS / SeRF** |
|---:|---:|---|---|---:|
| 1%  | 49   | 25.4 (α=256, r=0.967) | 623.6 (α=8, 1.1/16, r=0.925) | **24.6×** |
| 5%  | 249  | 142.8 (α=32, r=0.903) | 506.4 (α=2, 1.7/16, r=0.986) | **3.55×** |
| 10% | 498  | 154.3 (α=32, r=0.941) | 323.3 (α=1, 2.6/16, r=0.980) | **2.10×** |
| 25% | 1247 | 300.8 (α=16, r=0.908) | 196.1 (α=1, 5.0/16, r=0.994) | 0.65× |
| 50% | 2494 | 281.8 (α=16, r=0.919) | 119.8 (α=1, 9.0/16, r=0.990) | 0.43× |

(For reference: `ExactScan` ≈ 6.5k–8k QPS, `RangeFirstScan` 43.6k → 8.3k QPS as
s_B grows. See §6 for why these dominate in absolute terms.)

**The mechanism, confirmed.** At narrow B, the query window overlaps ~1 of 16
buckets, so Multi-SeRF skips ~15/16 of the data and reaches recall 0.9 with a
tiny over-fetch (α=8 → 1). SeRF+ResidualB, with no B index, must fetch a huge
A-neighbourhood (α=256 = 2560 of 5000 points) before residual-filtering B down
to the ~49 survivors — slow. As B widens, Multi-SeRF visits more buckets
(1.1 → 9.0) and the per-bucket search overhead accumulates, so the advantage
erodes and inverts past s_B ≈ 13%. Note Multi-SeRF's recall is *higher* than the
baseline's at every row: it reaches the 0.9 bar early and overshoots, because it
searches a smaller, on-target candidate set.

---

## 2. The bucket-count K trade-off (the real story)

Bigger K = finer B-partition = more buckets skipped at narrow B, but more
independent graph searches at wide B. Sweeping K (full A range, otherwise as §1):

**CS / SeRF+ResidualB QPS ratio, by K and s_B**

| s_B | K=4 | K=16 | K=32 |
|---:|---:|---:|---:|
| 1%  | 4.03× | 24.60× | **32.81×** |
| 5%  | 3.98× | 3.55×  | 3.22× |
| 10% | 3.79× | 2.10×  | 2.06× |
| 25% | 1.27× | 0.65×  | 0.50× |
| 50% | 0.82× | 0.43×  | 0.31× |

(`results_partB_K4.json`, `results_partB_main.json`, `results_partB_K32.json`.)

- **Narrow B favours large K.** K=32 peaks at 32.8× (s_B=1%); K=4 only 4.0×,
  because with 4 wide buckets the one overlapping bucket still holds ~1250 points
  and Multi-SeRF must over-fetch heavily (α=64) and residual-filter inside it.
- **Wide B favours small K.** At s_B=50%, K=4 holds at 0.82× (nearly matching the
  baseline — it runs only ~3 searches) while K=32 drops to 0.31× (it runs ~17).
- **The crossover (ratio = 1) moves with K:** ≈ 40% for K=4, ≈ 13% for K=16/32.

So **K is the design lever** between the two success criteria below. No single K
is best everywhere; the right K depends on the expected B-selectivity of the
workload.

---

## 3. Restrictive A range (A and B both filtered)

`results_partB_arestrict.json`. Same as §1 but the A predicate also has 25%
selectivity, so both the bucket routing (B) and the segment graph's A-filtering
are exercised. K=16.

| s_B | pts pass | SeRF+ResidualB QPS @0.9 (α, recall) | Multi-SeRF QPS @0.9 (α, buckets, recall) | **CS / SeRF** |
|---:|---:|---|---|---:|
| 1%  | 13  | 32.9 (α=256, r=0.877 ✗) | 559.6 (α=16, 1.1/16, r=0.931) | **17.0×** |
| 5%  | 63  | 78.1 (α=64, r=0.917)    | 567.7 (α=2, 1.7/16, r=0.977)  | **7.27×** |
| 10% | 127 | 106.2 (α=32, r=0.920)   | 300.5 (α=1, 2.6/16, r=0.986)  | **2.83×** |
| 25% | 315 | 224.7 (α=16, r=0.910)   | 213.4 (α=1, 5.0/16, r=0.997)  | 0.95× |
| 50% | 624 | 258.3 (α=16, r=0.925)   | 116.1 (α=1, 9.0/16, r=0.993)  | 0.45× |

The pattern holds with A-filtering active. Notably at s_B=1% the baseline
**cannot reach recall 0.9 even at α=256** (it tops out at 0.877): with both
predicates selective, residual-B filtering throws away too much. Multi-SeRF
reaches 0.931 *and* is 17× faster — the clearest single illustration of SeRF's
L1 limitation that the proposal set out to fix.

---

## 4. Against the proposal's success criteria

The proposal calls the project successful if **either** holds:

> **(1)** Multi-SeRF ≥ 2× QPS over SeRF+ResidualB at `s_B ≤ 5%`, recall floor 0.9.
> **(2)** Multi-SeRF within 20% of SeRF+ResidualB QPS at `s_B ≥ 50%`.

| | criterion (1) — narrow B | criterion (2) — wide B (ratio ≥ 0.83) |
|---|---|---|
| K=4  | ✅ 4.0× @1%, 4.0× @5% | ⚠️ 0.82× @50% — at the threshold |
| K=16 | ✅ 24.6× @1%, 3.6× @5% | ❌ 0.43× @50% |
| K=32 | ✅ 32.8× @1%, 3.2× @5% | ❌ 0.31× @50% |
| K=16, A=25% | ✅ 17× @1%, 7.3× @5% | ❌ 0.45× @50% |

- **Criterion (1) is met decisively in every configuration tested**, including
  with a restrictive A predicate. The recall floor of 0.9 is satisfied by
  Multi-SeRF at the reported points (0.925–0.997).
- **Criterion (2) is met only at small K (K=4, and only just).** At K≥16 it
  fails clearly: at full bucket coverage Multi-SeRF runs K independent graph
  searches against the baseline's one, so it is 2–3× *slower* at s_B=50%.

Since the proposal requires only one criterion, it is a **success by its own
bar (criterion 1)** — and honestly so, the narrow-B win is large and robust.
But the prototype also shows criterion (2) is *not* free: it holds only if K is
kept small, which blunts the narrow-B win. The proposal's framing ("at wide B,
Multi-SeRF approaches SeRF+ResidualB because the residual cost shrinks") is
incomplete — it omits that Multi-SeRF's own per-bucket search cost *grows* with
coverage. A faithful SeRF-2D core (one range-aware search instead of K) would
not pay this K-fold penalty; that is the gap between this prototype and the
full design.

---

## 5. Index size and build

Estimated serialised size (12 B/edge + vectors + attributes) and build time:

| index | edges | size | build (s) |
|---|---:|---:|---:|
| SeRF+ResidualB (K=1) | 159,728 | 2.60 MB | ~5.9 |
| Multi-SeRF K=4  | 158,912 | 2.59 MB | 4.82 |
| Multi-SeRF K=16 | 155,648 | 2.55 MB | 3.55 |
| Multi-SeRF K=32 | 151,296 | 2.50 MB | 2.41 |

Bucketing does **not** blow up storage — total edges are roughly constant (each
point keeps ~M neighbours regardless of partitioning), in fact slightly *fewer*
at larger K (no cross-bucket edges), and build is *faster* at larger K (K small
graphs of size n/K cost less than one graph of size n). This is consistent with
the proposal's "near-linear space" aspiration: the second attribute is indexed
by bucket containment, which is nearly free in space.

---

## 6. Why exact scan beats the graph methods here (scale caveat)

In absolute QPS, `RangeFirstScan` (vectorised NumPy: mask, then exact distance
on survivors) beats *every* graph method at *every* selectivity in these runs —
e.g. at s_B=1%, 43.6k QPS vs Multi-SeRF's 624 and SeRF+ResidualB's 25. This is
**expected and not a defeat of the idea**:

- n=5,000 is tiny. Brute-forcing even the 2,494 survivors at s_B=50% is a single
  vectorised NumPy call; the graph methods traverse node-by-node in interpreted
  Python. The comparison is dominated by NumPy-vs-Python, not by algorithm.
- Graph ANN earns its keep only when n is large enough that an O(n) scan is the
  bottleneck — SeRF reports at n=1M–10M. This prototype is a *mechanism* check,
  not a scale benchmark.

This is the same discipline as the Part A prototype: do not read absolute
latency across implementations; read the **like-for-like ratio** (Multi-SeRF vs
SeRF+ResidualB, identical code modulo bucketing).

---

## 7. What this prototype proves vs. does not

### Reasonably supported

- The Compound Segment **mechanism works**: B-bucket routing + per-bucket A
  search returns correct results and, at narrow B, reaches recall 0.9 far more
  cheaply than residual-B filtering (criterion 1, decisively, across K and with
  A-filtering active).
- The **bucket-count K trade-off** is real and quantified: large K maximises the
  narrow-B win and pushes the crossover earlier; small K preserves wide-B parity.
- Bucketing is **near-free in space** and cheaper to build than the single graph.
- SeRF's **L1 limitation is reproduced concretely** (§3, s_B=1%: the baseline
  cannot even reach recall 0.9 by over-fetching).

### Not established

- **No faithful SeRF-2D core.** The A upper bound is a residual filter, and the
  per-bucket graph is single-layer. A real SeRF-2D would do one range-aware
  search per bucket and would not pay the wide-B K-fold penalty as sharply.
- **Absolute performance / scale.** Pure Python, n=5k; the graph methods lose to
  vectorised exact scan here (§6). Nothing is shown at the n=1M+ regime where the
  approach is meant to matter.
- **Real data.** Synthetic vectors with attributes independent of the vectors —
  the easy case. Correlated attributes (e.g. price ↔ rating) would stress the
  equal-frequency bucketing and are untested.
- **No DBMS / SQL / persistence / updates** (the proposal's broader scope and its
  own L2 append-only limitation).

---

## 8. Files

- `design_notes.md` — architecture, simplified segment graph, isolation argument, scope
- `multiserf_proto.py` — `SegmentGraph1D` + `CompoundSegment` + ground truth/recall
- `run_experiments_partB.py` — runner (B-selectivity sweep, QPS-at-recall-0.9)
- `results_partB_main.json` / `` — headline (K=16, full A)
- `results_partB_K4.json`, `results_partB_K32.json` + logs — K sensitivity
- `results_partB_arestrict.json` / `` — restrictive A (25%)
- `results_partB_smoke.json` — n=1k smoke
- `README.md` — orientation + how to run
- `results.md` — this file

---

## 9. Post-hoc robustness checks (added 2026-07-05)

Three checks run after the write-up above; the sections above are unchanged.
All use the main config (n=5000, dim=32, nq=100, K=16, full A) unless noted.

### 9.1 Data-seed variance

CS / SeRF+ResidualB QPS ratio, main config, three data seeds (query seed is
independent and fixed; only the dataset changes):

| s_B | seed 0 (recorded §1) | seed 1 | seed 2 |
|---:|---:|---:|---:|
| 1%  | 24.60× | 15.05× | 15.16× |
| 5%  | 3.55×  | 3.99×  | 3.53×  |
| 10% | 2.10×  | 2.53×  | 2.70×  |
| 25% | 0.65×  | 0.76×  | 0.73×  |
| 50% | 0.43×  | 0.40×  | 0.38×  |

The narrow-B cell is **grid-quantised**: the baseline clears recall 0.9 at
α=256 on seed 0 but at α=128 on seeds 1–2, and since the α grid doubles per
step, the ratio jumps ~2× on that boundary. The honest headline is therefore
**15–25× at s_B=1%** (and a stable 3.5–4.0× at 5%), not the single-seed 24.6×.
Criterion (1) (≥2× at s_B≤5%) holds on every seed; the wide-B rows are stable
(0.38–0.43× at 50%). `results_partB_seed1.json`, `results_partB_seed2.json`.

### 9.2 Correlated B (the harder case for bucketing)

§7 flagged independent attributes as the easy case. This run makes B strongly
dependent on the vectors: Gaussian copula on the first vector coordinate,
rank-transformed back to a Uniform[0,1] marginal (`--b-corr 0.8`; realised rank
correlation ≈ 0.78). Equal-frequency bucketing sees the same marginal; what
changes is that B-buckets now map to regions of vector space.

| s_B | 1% | 5% | 10% | 25% | 50% |
|---|---:|---:|---:|---:|---:|
| CS / SeRF ratio | 14.49× | 3.82× | 2.76× | 0.76× | 0.41× |

Everything sits inside the seed-variance envelope of §9.1 — at this
correlation strength there is **no evidence that bucket routing degrades**
(recall at the reported points: 0.96–1.00). This softens, but does not remove,
the caveat: one correlation pattern, one strength, still synthetic.
`results_partB_bcorr.json`.

### 9.3 Scale check: n = 20,000 (single run, nq=50)

| s_B | SeRF+ResidualB QPS (α) | Multi-SeRF QPS (α, buckets) | ratio |
|---:|---|---|---:|
| 1%  | 24.0 (α=256) | 370.9 (α=16, 1.2/16) | **15.45×** |
| 5%  | 20.9 (α=128) | 234.5 (α=2, 1.8/16)  | **11.24×** |
| 10% | 27.4 (α=128) | 177.9 (α=1, 2.6/16)  | **6.49×** |
| 25% | 37.9 (α=64)  | 109.4 (α=1, 5.0/16)  | **2.89×** |
| 50% | 63.8 (α=32)  | 53.9 (α=1, 9.0/16)   | 0.85× |

At 4× the data, the advantage **grows across the board**: the crossover moves
past s_B=25%, and at 50% the ratio rises from 0.43× to 0.85× — nominally above
criterion (2)'s 0.83 bar, though from a single nq=50 run, so read it as "at
the threshold", not established. The mechanism is visible in the α columns:
the baseline's required over-fetch grows with n (α=128 at s_B=5%, vs 32 at
n=5k) while Multi-SeRF's stays flat (α=2). This is consistent with §6's scale
argument — bucketing pays off more as n grows — and with SeRF's own regime
(n=1M+), which remains untested here. Absolute-scan context at n=20k:
`RangeFirstScan` drops to 735 QPS at s_B=50% vs the graph methods' 54–64,
i.e. the brute-force gap is also closing with n. `results_partB_n20k.json`.

### Files added by §9

- `results_partB_seed1.json` / ``, `results_partB_seed2.json` /
  `` — seed variance
- `results_partB_bcorr.json` / `` — correlated B (ρ=0.8)
- `results_partB_n20k.json` / `` — n=20k scale check

---

## 10. Adaptive routing: resolving the K trade-off at query time (added 2026-07-05)

§2 concluded that no single K wins everywhere. But the trade-off does not have
to be resolved at build time. `AdaptiveIndex` keeps **both** the K=1 baseline
graph and the K=16 Compound Segment (~2× the single graph's edges; vectors
shareable) and picks an arm per query: equal-frequency bucketing makes a
B-selectivity estimate free — sum the covered fraction of each overlapping
bucket's B-span — and queries with estimated `s_B ≤ τ` (τ=0.15 here) go to the
bucketed index, the rest to the single graph.

`run_adaptive.py` / `results_partB_adaptive.json` (main config; all three arms
measured in the same process, so ratios are like-for-like):

| s_B | CS/SeRF (fixed K=16) | **Adaptive/SeRF** | routed to buckets | adaptive recall |
|---:|---:|---:|---:|---:|
| 1%  | 28.44× | **27.29×** | 100% | 0.925 |
| 5%  | 3.39×  | **3.29×**  | 100% | 0.986 |
| 10% | 3.01×  | **2.14×**  | 100% | 0.980 |
| 25% | 0.85×  | **1.07×**  | 0%   | 0.908 |
| 50% | 0.45×  | **1.00×**  | 0%   | 0.919 |

- Where bucketing wins (1–10%), adaptive tracks fixed K=16 (the small gap is
  routing/measurement noise — it runs the identical arm).
- Where bucketing loses (25–50%), adaptive routes everything to the K=1 graph
  and sits at parity instead of 0.45–0.85×.
- **Both of the proposal's success criteria now hold simultaneously**: ≥2× at
  `s_B ≤ 5%` (27×/3.3×) *and* within 20% of the baseline at `s_B ≥ 50%`
  (1.00×) — criterion (2) previously failed at every fixed K ≥ 16 (§4).

Caveats: single run, one seed, one τ; τ is workload- and scale-dependent (the
crossover moves with n, §9.3), so a production system would calibrate it. The
CS/SeRF ratios differ slightly from §1 (28.4× vs 24.6× at 1%) — same code and
data, different process/run; this is ordinary timing variance at the α-grid
boundary, see §9.1.

Note on the routing estimate: summing covered bucket-span fractions assumes B
is roughly uniform within each bucket, which holds here by construction. For
skewed B, per-bucket counts would give the exact answer at the same O(K) cost.

---

## 11. Real vectors: SIFT10K (added 2026-07-05)

`--dataset siftsmall`: the TEXMEX `siftsmall` corpus — 10,000 real 128-d SIFT
descriptors plus the corpus's own 100 held-out query vectors. The corpus has
no structured attributes, so a and b stay synthetic Uniform[0,1]: this run
makes the **vectors** real, not the attribute distribution.

**First attempt (default M=16, ef_build=64) failed the recall floor.** On real,
clustered vectors neither arm reached recall 0.9 at any tested α — both
plateaued at ≈0.66–0.75 (`results_partB_sift.json`). The simplified flat graph
that comfortably clears 0.9 on isotropic Gaussians does not have enough
connectivity for SIFT's cluster structure at M=16. Kept and reported because it
is a real limitation of `SegmentGraph1D`, discovered only by leaving synthetic
data. (Multi-SeRF still dominated — higher recall *and* ~14× the QPS — but
"QPS at recall 0.9" was not measurable.)

**With a stronger build (M=32, ef_build=200) the synthetic story reproduces**
(`results_partB_sift_M32.json` / ``, both arms ≥0.9 everywhere):

| s_B | SeRF+ResidualB QPS (α, recall) | Multi-SeRF QPS (α, buckets, recall) | ratio |
|---:|---|---|---:|
| 1%  | 20.0 (α=128, r=0.929) | 294.8 (α=8, 1.2/16, r=0.937) | **14.74×** |
| 5%  | 65.6 (α=32, r=0.947)  | 237.0 (α=2, 1.8/16, r=0.977) | **3.61×** |
| 10% | 229.7 (α=16, r=0.938) | 280.6 (α=1, 2.6/16, r=0.975) | 1.22× |
| 25% | 258.4 (α=16, r=0.927) | 113.4 (α=1, 5.0/16, r=0.983) | 0.44× |
| 50% | 138.6 (α=16, r=0.930) | 57.4 (α=1, 9.1/16, r=0.979)  | 0.41× |

- The headline mechanism **transfers to real vectors**: 14.7× at 1%, 3.6× at
  5%, criterion (1) met. The crossover sits slightly earlier (~10–12%) than on
  synthetic data at the same n-per-bucket.
- Honest scope: real vectors, synthetic independent attributes, n=10k, one
  run; and the result needed graph parameters retuned (M=32) — a sensitivity a
  production system would have to manage.

---

## 12. Scale: n = 100,000 (added 2026-07-05)

`results_partB_n100k.json` / `` (nq=50, K=16, single run;
build: 178 s for K=1, 148 s for K=16, ~52 MB estimated size either way).

| s_B | SeRF+ResidualB QPS (α, recall) | Multi-SeRF QPS (α, buckets, recall) | ratio |
|---:|---|---|---:|
| 1%  | 3.6 (α=512, r=0.888 ✗) | 80.9 (α=32, 1.2/16, r=0.936) | **22.5×** |
| 5%  | 3.6 (α=512, r=0.892 ✗) | 90.4 (α=16, 1.8/16, r=0.916) | **25.1×** |
| 10% | 3.6 (α=512, r=0.930)   | 63.0 (α=16, 2.6/16, r=0.930) | **17.3×** |
| 25% | 3.5 (α=512, r=0.904)   | 30.1 (α=16, 5.0/16, r=0.942) | **8.6×** |
| 50% | 3.8 (α=512, r=0.916)   | 18.9 (α=16, 9.0/16, r=0.944) | **5.0×** |

At 20× the headline dataset, the picture changes qualitatively:

- **The baseline is effectively broken at this scale.** It needs the α-grid
  maximum (512, i.e. fetch 5,120 candidates for k=10) in every cell, lands at
  3.5–3.8 QPS, and at s_B=1–5% still falls short of the 0.9 floor (0.888/0.892
  reported at its best point — so those two ratios *understate* the true
  at-recall gap).
- **The wide-B crossover is gone**: Multi-SeRF wins at every selectivity,
  including 5.0× at s_B=50%. The per-bucket graphs (6,250 points each) reach
  the recall floor with far smaller ef than one 100k-node graph — the
  wide-B penalty seen at n=5k (§1–2) is a small-n artifact of this
  implementation, not a property that worsens with scale.
- Trend across n for the 1% cell: 15–25× (5k) → 15.5× (20k) → 22.5× (100k);
  and for the 50% cell: 0.38–0.43× (5k) → 0.85× (20k) → 5.0× (100k).

Caveats: single run, nq=50, α grid capped at 512 (the baseline might clear the
floor at higher α, but only by getting slower); both arms share the same
ef = max(α·k, 64) coupling, so the comparison stays like-for-like. Still pure
single-thread Python and still below SeRF's n=1M+ regime.

**Engineering note (why not just make it faster):** an attempt to NumPy-
vectorise the per-node edge filter in `SegmentGraph1D.query` made queries
**2.4× slower** (ef=2560, n=5k): with M=16 a node has ~32 out-edges, and NumPy
per-call overhead on 32-element arrays dwarfs the tight-loop cost it replaces.
The change was verified behaviour-identical (bit-identical smoke results) and
then reverted. Pushing past n≈100k needs a compiled kernel (numba/C++), which
is outside this prototype's scope by design.

### Files added by §10–12

- `run_adaptive.py`, `results_partB_adaptive.json` / ``
- `results_partB_sift.json` / `` — SIFT10K, default build (recall
  floor not reached; kept as the negative finding)
- `results_partB_sift_M32.json` / `` — SIFT10K, M=32/ef=200
- `results_partB_n100k.json` / `` — n=100k scale run
- `demo.py`, `sql_demo.py` — narrated demo and DuckDB scalar-UDF demo
