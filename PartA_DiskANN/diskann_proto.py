"""
DiskANN-style on-disk ANN index prototype.

Standalone implementation outside DuckDB. Exercises the algorithmic core of the
Part A proposal (PQ-pruned Beam Search over a Vamana graph, with full-precision
rerank, against a page-aligned binary file) so its behaviour can be measured;
whether the proposal's specific recall / latency targets are met is the subject
of `results.md`, not a claim of this file. The future DuckDB extension would
replace `IndexReader._read_graph_page` with a `BufferManager::Pin` call on the
same byte layout.
"""

from __future__ import annotations

import heapq
import mmap
import os
import struct
import time
from dataclasses import dataclass

# Use perf_counter for all timing (monotonic, higher resolution than time.time)
_now = time.perf_counter

import numpy as np
from sklearn.cluster import KMeans

PAGE_SIZE = 4096
HEADER_SIZE = 96
MAGIC = 0x44414E50  # "DANP"
VERSION = 1
HEADER_STRUCT = "<IIQIIIIIQQQQ"


def _align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _l2sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    return np.einsum("...i,...i->...", diff, diff)


# ----------------------------------------------------------------------
# PQ
# ----------------------------------------------------------------------

@dataclass
class PQCodebook:
    pq_m: int
    pq_ks: int
    sub_dim: int
    centroids: np.ndarray  # (pq_m, pq_ks, sub_dim) float32

    @classmethod
    def train(cls, X: np.ndarray, pq_m: int, pq_ks: int = 256,
              sample_size: int = 50_000, seed: int = 0) -> "PQCodebook":
        n, dim = X.shape
        assert dim % pq_m == 0, f"dim {dim} not divisible by pq_m {pq_m}"
        sub_dim = dim // pq_m
        if n > sample_size:
            rng = np.random.default_rng(seed)
            X = X[rng.choice(n, sample_size, replace=False)]
        centroids = np.zeros((pq_m, pq_ks, sub_dim), dtype=np.float32)
        for j in range(pq_m):
            sub = X[:, j * sub_dim:(j + 1) * sub_dim]
            km = KMeans(n_clusters=pq_ks, n_init=3, random_state=seed + j,
                        max_iter=25)
            km.fit(sub)
            centroids[j] = km.cluster_centers_.astype(np.float32)
        return cls(pq_m, pq_ks, sub_dim, centroids)

    def encode(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        codes = np.zeros((n, self.pq_m), dtype=np.uint8)
        for j in range(self.pq_m):
            sub = X[:, j * self.sub_dim:(j + 1) * self.sub_dim]
            # distance to each centroid in this subquantizer
            # broadcasting: (n,1,sub_dim) - (1,ks,sub_dim) → (n,ks)
            d = np.einsum(
                "ni,ni->n", sub, sub)[:, None] \
                + np.einsum("ki,ki->k", self.centroids[j], self.centroids[j])[None, :] \
                - 2 * sub @ self.centroids[j].T
            codes[:, j] = d.argmin(axis=1).astype(np.uint8)
        return codes

    def asym_table(self, q: np.ndarray) -> np.ndarray:
        """Squared L2 between query subvectors and each centroid.
        Returns (pq_m, pq_ks) float32."""
        tbl = np.zeros((self.pq_m, self.pq_ks), dtype=np.float32)
        for j in range(self.pq_m):
            qs = q[j * self.sub_dim:(j + 1) * self.sub_dim]
            diff = self.centroids[j] - qs
            tbl[j] = np.einsum("ki,ki->k", diff, diff)
        return tbl

    def score(self, codes: np.ndarray, tbl: np.ndarray) -> np.ndarray:
        """codes: (n, pq_m) uint8, tbl: (pq_m, pq_ks). Returns (n,) float32."""
        # Sum across subquantizers using fancy indexing.
        # tbl[j, codes[:, j]] for each j, accumulate.
        out = np.zeros(codes.shape[0], dtype=np.float32)
        for j in range(self.pq_m):
            out += tbl[j, codes[:, j]]
        return out


# ----------------------------------------------------------------------
# Vamana builder (in-memory)
# ----------------------------------------------------------------------

def _greedy_search(X: np.ndarray, graph: list[list[int]], q: np.ndarray,
                   ep: int, ef: int) -> list[tuple[float, int]]:
    """Return up to `ef` visited candidates sorted by distance ascending.
    Standard greedy graph traversal used during Vamana build."""
    visited = {ep}
    # min-heap of unexpanded
    d_ep = float(_l2sq(q, X[ep]))
    candidates: list[tuple[float, int]] = [(d_ep, ep)]
    heapq.heapify(candidates)
    # max-heap (negated) of best results so far
    results: list[tuple[float, int]] = [(-d_ep, ep)]
    heapq.heapify(results)
    while candidates:
        d_c, c = heapq.heappop(candidates)
        worst = -results[0][0]
        if d_c > worst and len(results) >= ef:
            break
        for v in graph[c]:
            if v in visited:
                continue
            visited.add(v)
            d_v = float(_l2sq(q, X[v]))
            if len(results) < ef or d_v < -results[0][0]:
                heapq.heappush(candidates, (d_v, v))
                heapq.heappush(results, (-d_v, v))
                if len(results) > ef:
                    heapq.heappop(results)
    out = sorted((-d, v) for d, v in results)
    return out


def _robust_prune(p_id: int, cands: list[tuple[float, int]], X: np.ndarray,
                  alpha: float, M: int) -> list[int]:
    """Vamana α-pruning. `cands` is (dist_to_p, node_id) ascending."""
    cands = list(cands)
    selected: list[int] = []
    while cands and len(selected) < M:
        d_star, p_star = cands.pop(0)
        if p_star == p_id:
            continue
        selected.append(p_star)
        if not cands:
            break
        # remove any cand c such that alpha * d(p_star, c) <= d(p, c)
        ps_vec = X[p_star]
        rest_ids = np.array([c for _, c in cands], dtype=np.int64)
        d_ps_rest = _l2sq(ps_vec[None, :], X[rest_ids])
        d_p_rest = np.array([d for d, _ in cands], dtype=np.float32)
        keep_mask = (alpha * d_ps_rest) > d_p_rest
        cands = [cands[i] for i in range(len(cands)) if keep_mask[i]]
    return selected


def _vamana_pass(X: np.ndarray, graph: list[list[int]], ep: int,
                 M: int, ef: int, alpha: float, order: np.ndarray,
                 t0: float, verbose: bool, tag: str) -> None:
    n = X.shape[0]
    for i, p in enumerate(order):
        p = int(p)
        V = _greedy_search(X, graph, X[p], ep, ef)
        # merge in existing neighbors (with their distances to p)
        existing = [v for v in graph[p] if v != p]
        if existing:
            ex_ids = np.array(existing, dtype=np.int64)
            ex_d = _l2sq(X[p][None, :], X[ex_ids]).tolist()
            seen = {v for _, v in V}
            for vid, dd in zip(existing, ex_d):
                if vid not in seen:
                    V.append((dd, vid))
                    seen.add(vid)
            V.sort()
        V = [(d, v) for d, v in V if v != p]
        new_nbrs = _robust_prune(p, V, X, alpha, M)
        graph[p] = new_nbrs
        for q in new_nbrs:
            if p in graph[q]:
                continue
            graph[q].append(p)
            if len(graph[q]) > M:
                qv = X[q]
                qids = np.array(graph[q], dtype=np.int64)
                d_qq = _l2sq(qv[None, :], X[qids])
                cands_q = sorted(zip(d_qq.tolist(), qids.tolist()))
                graph[q] = _robust_prune(q, cands_q, X, alpha, M)
        if verbose and (i + 1) % max(1, n // 10) == 0:
            print(f"  vamana {tag}: {i+1}/{n} nodes ({_now()-t0:.1f}s)")


def build_vamana(X: np.ndarray, M: int = 32, ef_construction: int = 96,
                 alpha: float = 1.2, two_pass: bool = True, seed: int = 0,
                 verbose: bool = False) -> tuple[int, list[list[int]]]:
    """Vamana build. If two_pass=True, runs one pass with alpha=1.0 followed
    by another with the supplied alpha (>=1), matching the DiskANN paper's
    Section 3 procedure."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    centroid = X.mean(axis=0)
    ep = int(_l2sq(centroid[None, :], X).argmin())
    graph: list[list[int]] = [list(rng.choice(n, size=min(M, n - 1), replace=False))
                              for _ in range(n)]
    for i in range(n):
        graph[i] = [v for v in graph[i] if v != i]
    t0 = _now()
    if two_pass:
        order1 = rng.permutation(n)
        _vamana_pass(X, graph, ep, M, ef_construction, 1.0, order1,
                     t0, verbose, "α=1")
        order2 = rng.permutation(n)
        _vamana_pass(X, graph, ep, M, ef_construction, alpha, order2,
                     t0, verbose, f"α={alpha}")
    else:
        order = rng.permutation(n)
        _vamana_pass(X, graph, ep, M, ef_construction, alpha, order,
                     t0, verbose, f"α={alpha}")
    return ep, graph


# ----------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------

def _graph_record_size(M: int) -> int:
    # deg(u16) + pad(u16) + nbrs[M](u32)
    raw = 4 + 4 * M
    # cache-line align
    return _align_up(raw, 64)


def _nodes_per_page(M: int) -> int:
    rec = _graph_record_size(M)
    return max(1, PAGE_SIZE // rec)


def write_index(path: str, X: np.ndarray, ep: int, graph: list[list[int]],
                pq: PQCodebook, codes: np.ndarray, M: int) -> dict:
    n, dim = X.shape
    rec_size = _graph_record_size(M)
    npp = _nodes_per_page(M)
    graph_pages = (n + npp - 1) // npp
    graph_bytes = graph_pages * PAGE_SIZE

    pq_rows_per_page = max(1, PAGE_SIZE // pq.pq_m)
    pq_pages = (n + pq_rows_per_page - 1) // pq_rows_per_page
    pq_bytes = pq_pages * PAGE_SIZE

    full_rows_per_page = max(1, PAGE_SIZE // (4 * dim))
    full_pages = (n + full_rows_per_page - 1) // full_rows_per_page
    full_bytes = full_pages * PAGE_SIZE

    codebook_bytes = pq.centroids.nbytes
    codebook_bytes_padded = _align_up(codebook_bytes, PAGE_SIZE)

    graph_off = PAGE_SIZE
    pq_off = graph_off + graph_bytes
    full_off = pq_off + pq_bytes
    codebook_off = full_off + full_bytes
    total = codebook_off + codebook_bytes_padded

    with open(path, "wb") as f:
        f.truncate(total)
        # header
        f.seek(0)
        hdr = struct.pack(
            HEADER_STRUCT,
            MAGIC, VERSION, n, dim, M, ep, pq.pq_m, pq.pq_ks,
            graph_off, pq_off, full_off, codebook_off,
        )
        f.write(hdr)
        # zero-pad rest of page
        f.write(b"\x00" * (PAGE_SIZE - len(hdr)))

        # graph
        f.seek(graph_off)
        buf = bytearray(graph_bytes)
        for u in range(n):
            page = u // npp
            slot = u % npp
            off = page * PAGE_SIZE + slot * rec_size
            nbrs = graph[u][:M]
            deg = len(nbrs)
            struct.pack_into("<HH", buf, off, deg, 0)
            for j, v in enumerate(nbrs):
                struct.pack_into("<I", buf, off + 4 + 4 * j, v)
        f.write(bytes(buf))

        # pq codes (n, pq_m) row-major, packed page-aligned
        f.seek(pq_off)
        pq_buf = bytearray(pq_bytes)
        for u in range(n):
            page = u // pq_rows_per_page
            slot = u % pq_rows_per_page
            off = page * PAGE_SIZE + slot * pq.pq_m
            pq_buf[off:off + pq.pq_m] = bytes(codes[u])
        f.write(bytes(pq_buf))

        # full vectors (n, dim) float32 row-major
        f.seek(full_off)
        full_buf = bytearray(full_bytes)
        Xf = X.astype(np.float32, copy=False)
        row_bytes = 4 * dim
        for u in range(n):
            page = u // full_rows_per_page
            slot = u % full_rows_per_page
            off = page * PAGE_SIZE + slot * row_bytes
            full_buf[off:off + row_bytes] = Xf[u].tobytes()
        f.write(bytes(full_buf))

        # codebook
        f.seek(codebook_off)
        f.write(pq.centroids.tobytes())
        if codebook_bytes_padded > codebook_bytes:
            f.write(b"\x00" * (codebook_bytes_padded - codebook_bytes))

    return {
        "total_bytes": total,
        "graph_bytes": graph_bytes,
        "pq_bytes": pq_bytes,
        "full_bytes": full_bytes,
        "codebook_bytes": codebook_bytes,
        "rec_size": rec_size,
        "nodes_per_page": npp,
        "graph_pages": graph_pages,
        "pq_pages": pq_pages,
        "full_pages": full_pages,
    }


# ----------------------------------------------------------------------
# Reader / query
# ----------------------------------------------------------------------

class IndexReader:
    """Reads a prototype index file. Two access modes:
        mode='eager' — load all segments into numpy arrays at open time
                       (pure in-memory baseline)
        mode='mmap'  — keep segments mmap'd; each graph neighbor lookup is
                       a memoryview slice that may trigger a page fault.
    """

    def __init__(self, path: str, mode: str = "mmap"):
        assert mode in ("eager", "mmap")
        self.path = path
        self.mode = mode
        self._f = open(path, "rb")
        hdr = self._f.read(HEADER_SIZE)
        (magic, version, n, dim, M, ep, pq_m, pq_ks,
         graph_off, pq_off, full_off, codebook_off) = struct.unpack_from(HEADER_STRUCT, hdr, 0)
        if magic != MAGIC:
            raise ValueError(f"bad magic {magic:#x}")
        if version != VERSION:
            raise ValueError(f"unsupported version {version}")
        self.n = n
        self.dim = dim
        self.M = M
        self.ep = ep
        self.pq_m = pq_m
        self.pq_ks = pq_ks
        self.graph_off = graph_off
        self.pq_off = pq_off
        self.full_off = full_off
        self.codebook_off = codebook_off
        self.rec_size = _graph_record_size(M)
        self.npp = _nodes_per_page(M)
        self.pq_rows_per_page = max(1, PAGE_SIZE // pq_m)
        self.full_rows_per_page = max(1, PAGE_SIZE // (4 * dim))
        # codebook always pinned
        sub_dim = dim // pq_m
        self._f.seek(codebook_off)
        cb_bytes = pq_m * pq_ks * sub_dim * 4
        self.codebook = np.frombuffer(
            self._f.read(cb_bytes), dtype=np.float32
        ).reshape(pq_m, pq_ks, sub_dim).copy()
        self.sub_dim = sub_dim

        if mode == "eager":
            self._load_eager()
        else:
            self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def _load_eager(self):
        # graph
        self._f.seek(self.graph_off)
        gbytes = (self.n + self.npp - 1) // self.npp * PAGE_SIZE
        gbuf = self._f.read(gbytes)
        graph = [[] for _ in range(self.n)]
        for u in range(self.n):
            page = u // self.npp
            slot = u % self.npp
            off = page * PAGE_SIZE + slot * self.rec_size
            deg = struct.unpack_from("<H", gbuf, off)[0]
            nbrs = struct.unpack_from(f"<{deg}I", gbuf, off + 4)
            graph[u] = list(nbrs)
        self._graph_eager = graph
        # pq codes
        self._f.seek(self.pq_off)
        pbytes = (self.n + self.pq_rows_per_page - 1) // self.pq_rows_per_page * PAGE_SIZE
        pbuf = self._f.read(pbytes)
        codes = np.zeros((self.n, self.pq_m), dtype=np.uint8)
        for u in range(self.n):
            page = u // self.pq_rows_per_page
            slot = u % self.pq_rows_per_page
            off = page * PAGE_SIZE + slot * self.pq_m
            codes[u] = np.frombuffer(pbuf[off:off + self.pq_m], dtype=np.uint8)
        self._codes_eager = codes
        # full vectors
        self._f.seek(self.full_off)
        fbytes = (self.n + self.full_rows_per_page - 1) // self.full_rows_per_page * PAGE_SIZE
        fbuf = self._f.read(fbytes)
        full = np.zeros((self.n, self.dim), dtype=np.float32)
        rb = 4 * self.dim
        for u in range(self.n):
            page = u // self.full_rows_per_page
            slot = u % self.full_rows_per_page
            off = page * PAGE_SIZE + slot * rb
            full[u] = np.frombuffer(fbuf[off:off + rb], dtype=np.float32)
        self._full_eager = full

    def close(self):
        if self.mode == "mmap":
            self._mm.close()
        self._f.close()

    # -- page reads -----------------------------------------------------
    def _read_neighbors(self, u: int) -> np.ndarray:
        if self.mode == "eager":
            return np.asarray(self._graph_eager[u], dtype=np.int64)
        page = u // self.npp
        slot = u % self.npp
        off = self.graph_off + page * PAGE_SIZE + slot * self.rec_size
        deg = int.from_bytes(self._mm[off:off + 2], "little")
        raw = self._mm[off + 4:off + 4 + 4 * deg]
        return np.frombuffer(raw, dtype=np.uint32).astype(np.int64)

    def _read_pq_codes(self, ids: np.ndarray) -> np.ndarray:
        if self.mode == "eager":
            return self._codes_eager[ids]
        out = np.zeros((len(ids), self.pq_m), dtype=np.uint8)
        for i, u in enumerate(ids):
            u = int(u)
            page = u // self.pq_rows_per_page
            slot = u % self.pq_rows_per_page
            off = self.pq_off + page * PAGE_SIZE + slot * self.pq_m
            out[i] = np.frombuffer(self._mm[off:off + self.pq_m], dtype=np.uint8)
        return out

    def _read_full(self, u: int) -> np.ndarray:
        if self.mode == "eager":
            return self._full_eager[u]
        page = u // self.full_rows_per_page
        slot = u % self.full_rows_per_page
        rb = 4 * self.dim
        off = self.full_off + page * PAGE_SIZE + slot * rb
        return np.frombuffer(self._mm[off:off + rb], dtype=np.float32)

    def _full_dist_many(self, q: np.ndarray, ids: np.ndarray) -> np.ndarray:
        out = np.zeros(len(ids), dtype=np.float32)
        for i, u in enumerate(ids):
            fv = self._read_full(int(u))
            diff = fv - q
            out[i] = float(diff @ diff)
        return out

    # -- query ----------------------------------------------------------
    def beam_search(self, q: np.ndarray, k: int, L: int,
                    use_pq: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Returns (ids[k], exact_dists[k]) sorted ascending by exact L2.

        use_pq=False switches to full-precision distance during traversal
        (diagnostic mode — defeats the disk-savings of PQ but isolates the
        graph quality from the PQ approximation error)."""
        q = q.astype(np.float32)
        # asymmetric PQ table
        tbl = np.zeros((self.pq_m, self.pq_ks), dtype=np.float32)
        for j in range(self.pq_m):
            qs = q[j * self.sub_dim:(j + 1) * self.sub_dim]
            diff = self.codebook[j] - qs
            tbl[j] = np.einsum("ki,ki->k", diff, diff)

        def pq_dist_many(ids: np.ndarray) -> np.ndarray:
            if not use_pq:
                return self._full_dist_many(q, ids)
            codes = self._read_pq_codes(ids)
            d = np.zeros(len(ids), dtype=np.float32)
            for j in range(self.pq_m):
                d += tbl[j, codes[:, j]]
            return d

        visited = np.zeros(self.n, dtype=bool)
        ep = self.ep
        d_ep = float(pq_dist_many(np.array([ep], dtype=np.int64))[0])
        # min-heap of unexpanded candidates
        candidates: list[tuple[float, int]] = [(d_ep, ep)]
        heapq.heapify(candidates)
        # max-heap (negated) of current pool, size <= L
        pool: list[tuple[float, int]] = [(-d_ep, ep)]
        heapq.heapify(pool)
        visited[ep] = True

        while candidates:
            d_c, c = heapq.heappop(candidates)
            worst = -pool[0][0]
            if len(pool) >= L and d_c > worst:
                break
            nbrs = self._read_neighbors(c)
            if len(nbrs) == 0:
                continue
            unvisited_mask = ~visited[nbrs]
            unvisited = nbrs[unvisited_mask]
            if len(unvisited) == 0:
                continue
            visited[unvisited] = True
            dists = pq_dist_many(unvisited)
            for v, dv in zip(unvisited.tolist(), dists.tolist()):
                if len(pool) < L or dv < -pool[0][0]:
                    heapq.heappush(candidates, (dv, v))
                    heapq.heappush(pool, (-dv, v))
                    if len(pool) > L:
                        heapq.heappop(pool)

        # rerank top-L by exact L2 on full vectors
        top = sorted((-d, v) for d, v in pool)
        ids = np.array([v for _, v in top], dtype=np.int64)
        exact = np.zeros(len(ids), dtype=np.float32)
        for i, u in enumerate(ids):
            fv = self._read_full(int(u))
            diff = fv - q
            exact[i] = float(diff @ diff)
        order = np.argsort(exact)
        return ids[order][:k], exact[order][:k]


# ----------------------------------------------------------------------
# High-level helpers
# ----------------------------------------------------------------------

def build_and_write(X: np.ndarray, path: str, M: int = 32,
                    ef_construction: int = 64, pq_m: int = 8,
                    pq_ks: int = 256, alpha: float = 1.2,
                    seed: int = 0, verbose: bool = False) -> dict:
    """End-to-end: train PQ, build Vamana graph, write file. Returns stats."""
    n, dim = X.shape
    stats: dict = {"n": n, "dim": dim, "M": M,
                   "ef_construction": ef_construction,
                   "pq_m": pq_m, "pq_ks": pq_ks, "alpha": alpha}
    t0 = _now()
    pq = PQCodebook.train(X, pq_m=pq_m, pq_ks=pq_ks, seed=seed)
    stats["t_pq_train"] = _now() - t0
    t0 = _now()
    codes = pq.encode(X)
    stats["t_pq_encode"] = _now() - t0
    t0 = _now()
    ep, graph = build_vamana(X, M=M, ef_construction=ef_construction,
                             alpha=alpha, seed=seed, verbose=verbose)
    stats["t_build_graph"] = _now() - t0
    t0 = _now()
    layout = write_index(path, X, ep, graph, pq, codes, M)
    stats["t_write"] = _now() - t0
    stats.update(layout)
    stats["build_time_total"] = (stats["t_pq_train"] + stats["t_pq_encode"]
                                 + stats["t_build_graph"] + stats["t_write"])
    return stats


def exact_knn(X: np.ndarray, Q: np.ndarray, k: int) -> np.ndarray:
    """Brute-force ground truth, returns ids of shape (nq, k)."""
    nq = Q.shape[0]
    out = np.zeros((nq, k), dtype=np.int64)
    # ||x||^2 depends only on X; compute once and reuse across batches.
    x_norm = np.einsum("ni,ni->n", X, X)
    batch = 64
    for i in range(0, nq, batch):
        q = Q[i:i + batch]
        q_norm = np.einsum("ni,ni->n", q, q)
        d = x_norm[None, :] + q_norm[:, None] - 2 * q @ X.T
        idx = np.argpartition(d, k, axis=1)[:, :k]
        sub_d = np.take_along_axis(d, idx, axis=1)
        order = np.argsort(sub_d, axis=1)
        out[i:i + batch] = np.take_along_axis(idx, order, axis=1)
    return out


def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    """pred, truth: (nq, k). Returns avg fraction of truth[:k] found in pred[:k]."""
    nq = pred.shape[0]
    hits = 0
    for i in range(nq):
        hits += len(set(pred[i, :k].tolist()) & set(truth[i, :k].tolist()))
    return hits / (nq * k)
