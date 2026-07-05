"""
Multi-SeRF prototype — Compound Segment over a simplified 1D segment graph.

Standalone implementation (no DBMS). Validates the Part B proposal's one
load-bearing claim: that partitioning on a secondary ordered attribute B and
searching only the buckets overlapping the B-range beats a single-attribute
SeRF index that filters B as a residual — most strongly when the B predicate
is selective.

Design (see design_notes.md):
  - `SegmentGraph1D`  : a single-layer, SeRF-inspired proximity graph over one
                        ordered attribute `a`. Edges carry validity intervals
                        [death_a, birth_a]; a query left-bound `a_lo` selects
                        the active sub-graph natively, the upper bound `a_hi`
                        is a residual accept-test during the walk.
  - `CompoundSegment` : partition by `b` into K equal-frequency buckets, one
                        SegmentGraph1D per bucket over `a`. K=1 degenerates to
                        the SeRF+ResidualB baseline, so the same code path
                        powers both arms and the comparison isolates bucketing.

Pure Python + NumPy. Latencies are overhead-bound; read ratios, not absolutes.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np

_now = time.perf_counter

# death sentinel for an edge that survived to the end of construction: a value
# strictly below any real attribute. Real attributes here live in [0, 1], so
# -1e18 is safely below. Used so the activity test `death < a_lo <= birth`
# treats never-pruned edges as always-active for any finite a_lo.
_DEATH_NEVER = -1e18


def _l2_to_many(q: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Squared L2 from one query vector q (d,) to each row of mat (n, d)."""
    diff = mat - q
    return np.einsum("ij,ij->i", diff, diff)


# ----------------------------------------------------------------------
# SegmentGraph1D
# ----------------------------------------------------------------------

class SegmentGraph1D:
    """SeRF-inspired segment graph over a single ordered attribute `a`.

    Built by inserting points in descending-`a` order into a flat proximity
    graph; each directed edge records the attribute interval over which it is
    part of the live graph. A query with left bound `a_lo` follows only edges
    active at `a_lo` and only visits nodes with `a >= a_lo` — i.e. it searches
    exactly the graph induced by the points satisfying the lower bound. The
    upper bound `a_hi` is applied as an accept-test (a visited node enters the
    result pool only if `a <= a_hi`); see design_notes.md §2 for why this
    asymmetry is acceptable and cancels out of the headline comparison.
    """

    def __init__(self, vecs: np.ndarray, a_vals: np.ndarray, ids: np.ndarray,
                 M: int = 16, ef: int = 64, seed: int = 0):
        self.vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        self.a = np.asarray(a_vals, dtype=np.float64)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.m = int(self.vecs.shape[0])
        self.M = M
        self.ef_build = ef
        self.a_min = float(self.a.min()) if self.m else 0.0
        self.a_max = float(self.a.max()) if self.m else 0.0
        self.entry = 0
        # edges[u] = list of (v_local, birth_a, death_a)
        self.edges: list[list[tuple[int, float, float]]] = [[] for _ in range(self.m)]
        self.n_edges = 0
        if self.m:
            self._build()

    # -- construction ---------------------------------------------------
    def _greedy_build(self, p: int, live: list[dict]) -> list[int]:
        """Greedy best-first search over the current live graph from the entry
        point; returns visited local ids sorted by distance ascending."""
        q = self.vecs[p]
        entry = self.entry
        d_e = float(_l2_to_many(q, self.vecs[entry:entry + 1])[0])
        visited = {entry}
        cand = [(d_e, entry)]
        beam = [(-d_e, entry)]
        ef = self.ef_build
        while cand:
            d, u = heapq.heappop(cand)
            if d > -beam[0][0]:
                break
            nb = [v for v in live[u].keys() if v not in visited]
            if not nb:
                continue
            for v in nb:
                visited.add(v)
            dd = _l2_to_many(q, self.vecs[nb])
            for v, dv in zip(nb, dd.tolist()):
                if len(beam) < ef or dv < -beam[0][0]:
                    heapq.heappush(cand, (dv, v))
                    heapq.heappush(beam, (-dv, v))
                    if len(beam) > ef:
                        heapq.heappop(beam)
        return [v for _, v in sorted((-d, v) for d, v in beam)]

    def _prune(self, v: int, live: list[dict], cur_a: float) -> None:
        """Cap v's live out-degree at M, keeping the M nearest; archive the
        evicted edges with death_a = cur_a (they die at this threshold)."""
        items = list(live[v].items())  # (w, birth)
        ws = [w for w, _ in items]
        dd = _l2_to_many(self.vecs[v], self.vecs[ws])
        keep_order = np.argsort(dd)[:self.M]
        keep = {ws[i] for i in keep_order.tolist()}
        for w, birth in items:
            if w not in keep:
                self.edges[v].append((w, birth, cur_a))
                self.n_edges += 1
                del live[v][w]

    def _build(self) -> None:
        m = self.m
        order = np.argsort(self.a, kind="stable")[::-1]  # descending a
        self.entry = int(order[0])
        live: list[dict] = [dict() for _ in range(m)]
        for step in range(m):
            p = int(order[step])
            a_p = float(self.a[p])
            if step == 0:
                continue
            neigh = self._greedy_build(p, live)[:self.M]
            for v in neigh:
                if v not in live[p]:
                    live[p][v] = a_p
                if p not in live[v]:
                    live[v][p] = a_p
                if len(live[v]) > self.M:
                    self._prune(v, live, a_p)
            if len(live[p]) > self.M:
                self._prune(p, live, a_p)
        # flush survivors with the never-pruned sentinel
        for u in range(m):
            for v, birth in live[u].items():
                self.edges[u].append((v, birth, _DEATH_NEVER))
                self.n_edges += 1

    # -- query ----------------------------------------------------------
    def query(self, q: np.ndarray, a_lo: float, a_hi: float, k: int,
              ef: int) -> tuple[list[int], list[float]]:
        """Range-filtered ANN: approximate k nearest of q among points with
        a in [a_lo, a_hi]. Returns (global_ids, squared_dists) ascending."""
        if self.m == 0 or a_lo > self.a_max or a_hi < self.a_min:
            return [], []
        if not math.isfinite(a_lo):
            a_lo = -1e9
        if not math.isfinite(a_hi):
            a_hi = 1e9
        q = np.asarray(q, dtype=np.float32)
        entry = self.entry
        d_e = float(_l2_to_many(q, self.vecs[entry:entry + 1])[0])
        visited = {entry}
        cand = [(d_e, entry)]
        beam = [(-d_e, entry)]            # best ef *traversed* — drives the walk
        accepted: list[tuple[float, int]] = []
        a = self.a
        if a_lo <= a[entry] <= a_hi:
            accepted.append((d_e, entry))
        while cand:
            d, u = heapq.heappop(cand)
            if d > -beam[0][0]:
                break
            nb = []
            for (v, birth, death) in self.edges[u]:
                if death < a_lo <= birth and v not in visited and a[v] >= a_lo:
                    nb.append(v)
            if not nb:
                continue
            for v in nb:
                visited.add(v)
            dd = _l2_to_many(q, self.vecs[nb])
            for v, dv in zip(nb, dd.tolist()):
                if len(beam) < ef or dv < -beam[0][0]:
                    heapq.heappush(cand, (dv, v))
                    heapq.heappush(beam, (-dv, v))
                    if len(beam) > ef:
                        heapq.heappop(beam)
                    if a[v] <= a_hi:      # a >= a_lo already guaranteed
                        accepted.append((dv, v))
        accepted.sort()
        out_ids = [int(self.ids[v]) for _, v in accepted[:k]]
        out_d = [float(d) for d, _ in accepted[:k]]
        return out_ids, out_d

    def size_bytes(self) -> int:
        """Estimated serialised footprint: each directed edge = 4B id + 4B
        birth + 4B death (float32) = 12B; plus per-node vector + attribute."""
        edge_bytes = self.n_edges * 12
        vec_bytes = self.vecs.nbytes
        attr_bytes = self.m * 8
        return edge_bytes + vec_bytes + attr_bytes


# ----------------------------------------------------------------------
# CompoundSegment
# ----------------------------------------------------------------------

@dataclass
class _Bucket:
    sg: SegmentGraph1D
    b_min: float
    b_max: float


class CompoundSegment:
    """Multi-SeRF's Compound Segment: partition the data by attribute `b` into
    K equal-frequency buckets, build one SegmentGraph1D per bucket over `a`.

    K=1 makes this exactly the SeRF+ResidualB baseline (one A-graph over all
    points, B applied as a residual filter), so the same code powers both arms
    of the comparison.
    """

    def __init__(self, vecs: np.ndarray, a: np.ndarray, b: np.ndarray,
                 K: int = 16, M: int = 16, ef: int = 64, seed: int = 0):
        self.K = K
        self.b = np.asarray(b, dtype=np.float64)
        n = int(vecs.shape[0])
        order = np.argsort(self.b, kind="stable")
        bounds = np.linspace(0, n, K + 1).astype(int)
        self.buckets: list[_Bucket] = []
        t0 = _now()
        for j in range(K):
            idx = order[bounds[j]:bounds[j + 1]]
            if len(idx) == 0:
                continue
            sg = SegmentGraph1D(vecs[idx], a[idx], idx, M=M, ef=ef, seed=seed)
            self.buckets.append(_Bucket(sg, float(self.b[idx].min()),
                                        float(self.b[idx].max())))
        self.build_s = _now() - t0

    def query(self, q: np.ndarray, a_lo: float, a_hi: float,
              b_lo: float, b_hi: float, k: int, alpha: float = 2.0,
              ef: int | None = None) -> tuple[list[int], list[float], int]:
        """Approximate k nearest of q with a in [a_lo,a_hi] and b in [b_lo,b_hi].
        Returns (global_ids, squared_dists, n_buckets_visited)."""
        kp = max(k, int(math.ceil(alpha * k)))
        ef_q = ef if ef is not None else max(kp, 32)
        cands: list[tuple[float, int]] = []
        visited_buckets = 0
        for bk in self.buckets:
            if bk.b_max < b_lo or bk.b_min > b_hi:   # disjoint — skip
                continue
            visited_buckets += 1
            ids, ds = bk.sg.query(q, a_lo, a_hi, kp, ef_q)
            interior = (bk.b_min >= b_lo and bk.b_max <= b_hi)
            if interior:
                cands.extend(zip(ds, ids))
            else:                                    # boundary — residual on b
                for gid, dd in zip(ids, ds):
                    if b_lo <= self.b[gid] <= b_hi:
                        cands.append((dd, gid))
        cands.sort()
        return ([g for _, g in cands[:k]],
                [d for d, _ in cands[:k]],
                visited_buckets)

    def size_bytes(self) -> int:
        return sum(bk.sg.size_bytes() for bk in self.buckets)

    def n_edges(self) -> int:
        return sum(bk.sg.n_edges for bk in self.buckets)


# ----------------------------------------------------------------------
# AdaptiveIndex
# ----------------------------------------------------------------------

class AdaptiveIndex:
    """Query-adaptive routing between a K=1 baseline graph and a K>1
    Compound Segment over the same data.

    The K trade-off (large K wins narrow B, small K wins wide B) does not have
    to be resolved at build time: equal-frequency bucketing makes a
    B-selectivity estimate free at query time. Each bucket holds ~n/K points,
    so summing the covered fraction of every overlapping bucket's B-span
    estimates s_B from index metadata alone — no data scan. Queries with
    estimated s_B <= tau route to the bucketed index, the rest to the single
    graph. Space cost: both indexes are kept (~2x edges; vectors shareable).

    tau is workload- and scale-dependent: the measured ratio=1 crossover is
    ~13% for K=16 at n=5k but moves past 25% at n=20k. Default 0.15 matches
    the n=5k headline setting.
    """

    def __init__(self, cs1: CompoundSegment, csK: CompoundSegment,
                 tau: float = 0.15):
        assert len(cs1.buckets) == 1, "cs1 must be the K=1 baseline"
        self.cs1 = cs1
        self.csK = csK
        self.tau = tau

    def estimate_sb(self, b_lo: float, b_hi: float) -> float:
        """Estimated fraction of points passing the B predicate, from bucket
        boundaries only (assumes B ~uniform within a bucket's span)."""
        K = len(self.csK.buckets)
        est = 0.0
        for bk in self.csK.buckets:
            if bk.b_max < b_lo or bk.b_min > b_hi:
                continue
            span = bk.b_max - bk.b_min
            if span <= 0.0:
                est += 1.0 / K
            else:
                cover = min(b_hi, bk.b_max) - max(b_lo, bk.b_min)
                est += min(1.0, max(0.0, cover / span)) / K
        return est

    def query(self, q: np.ndarray, a_lo: float, a_hi: float,
              b_lo: float, b_hi: float, k: int, alpha: float = 2.0,
              ef: int | None = None
              ) -> tuple[list[int], list[float], int, bool]:
        """Route to one arm; returns (ids, dists, buckets_visited, used_bucketed)."""
        use_bucketed = self.estimate_sb(b_lo, b_hi) <= self.tau
        arm = self.csK if use_bucketed else self.cs1
        ids, ds, vb = arm.query(q, a_lo, a_hi, b_lo, b_hi, k,
                                alpha=alpha, ef=ef)
        return ids, ds, vb, use_bucketed


# ----------------------------------------------------------------------
# Ground truth + metrics
# ----------------------------------------------------------------------

def exact_filtered_knn(vecs: np.ndarray, a: np.ndarray, b: np.ndarray,
                       q: np.ndarray, a_lo: float, a_hi: float,
                       b_lo: float, b_hi: float, k: int) -> list[int]:
    """Brute-force ground truth: exact k nearest among points satisfying both
    range predicates. Returns global ids ascending by distance (≤ k of them)."""
    mask = (a >= a_lo) & (a <= a_hi) & (b >= b_lo) & (b <= b_hi)
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    d = _l2_to_many(np.asarray(q, dtype=np.float32), vecs[idx])
    kk = min(k, idx.size)
    part = np.argpartition(d, kk - 1)[:kk]
    part = part[np.argsort(d[part])]
    return idx[part].tolist()


def recall_at_k(pred_ids: list[int], truth_ids: list[int], k: int) -> float:
    """Fraction of the true top-k recovered. If fewer than k points satisfy the
    predicate, recall is measured against that smaller ground-truth set."""
    if not truth_ids:
        return 1.0  # empty result set is trivially correct
    denom = min(k, len(truth_ids))
    hit = len(set(pred_ids[:k]) & set(truth_ids[:denom]))
    return hit / denom
