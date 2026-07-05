"""
Sanity tests for the Part A on-disk ANN prototype.

These pin the storage-layer invariants the write-up leans on:

  1. the page-aligned file round-trips *exactly* — graph neighbours, PQ codes,
     and full vectors read back byte-identical to what was written;
  2. eager and mmap are the same algorithm over the same bytes: identical ids
     and distances per query;
  3. the header/self-description is correct (magic, version, params, offsets
     page-aligned, segment sizes add up);
  4. `exact_knn` / `recall_at_k` — the measurement tools — are themselves
     correct.

Run:  py -3 -m pytest test_sanity.py -q      (~10 s: builds one small index)
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

import diskann_proto as dap

N, DIM, M, EF, PQ_M, PQ_KS = 300, 32, 8, 32, 4, 16


@pytest.fixture(scope="module")
def small_index(tmp_path_factory):
    """Build one small index with the low-level API so the in-memory graph,
    codes, and vectors are available to compare against what reads back."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, DIM)).astype(np.float32)
    Q = np.random.default_rng(12345).standard_normal((20, DIM)).astype(np.float32)
    pq = dap.PQCodebook.train(X, pq_m=PQ_M, pq_ks=PQ_KS, seed=0)
    codes = pq.encode(X)
    ep, graph = dap.build_vamana(X, M=M, ef_construction=EF, seed=0)
    path = str(tmp_path_factory.mktemp("idx") / "small.idx")
    stats = dap.write_index(path, X, ep, graph, pq, codes, M)
    return dict(X=X, Q=Q, pq=pq, codes=codes, ep=ep, graph=graph,
                path=path, stats=stats)


# ----------------------------------------------------------------------
# header / layout
# ----------------------------------------------------------------------

def test_header_fields_and_page_alignment(small_index):
    r = dap.IndexReader(small_index["path"], mode="eager")
    try:
        assert (r.n, r.dim, r.M) == (N, DIM, M)
        assert (r.pq_m, r.pq_ks) == (PQ_M, PQ_KS)
        assert r.ep == small_index["ep"]
        for off in (r.graph_off, r.pq_off, r.full_off, r.codebook_off):
            assert off % dap.PAGE_SIZE == 0
        assert r.graph_off == dap.PAGE_SIZE   # header owns exactly one page
    finally:
        r.close()


def test_segment_sizes_add_up(small_index):
    import os
    s = small_index["stats"]
    cb_padded = dap._align_up(s["codebook_bytes"], dap.PAGE_SIZE)
    assert s["total_bytes"] == (dap.PAGE_SIZE + s["graph_bytes"]
                                + s["pq_bytes"] + s["full_bytes"] + cb_padded)
    assert os.path.getsize(small_index["path"]) == s["total_bytes"]
    assert s["total_bytes"] % dap.PAGE_SIZE == 0


def test_bad_magic_is_rejected(small_index, tmp_path):
    corrupt = tmp_path / "corrupt.idx"
    data = bytearray(open(small_index["path"], "rb").read())
    struct.pack_into("<I", data, 0, 0xDEADBEEF)
    corrupt.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="bad magic"):
        dap.IndexReader(str(corrupt), mode="eager")


# ----------------------------------------------------------------------
# byte-exact roundtrip
# ----------------------------------------------------------------------

def test_roundtrip_graph_codes_vectors(small_index):
    # what the write-up calls "round-trips correctly" — checked byte-exact for
    # every node, in both access modes
    X, codes, graph = (small_index["X"], small_index["codes"],
                       small_index["graph"])
    for mode in ("eager", "mmap"):
        r = dap.IndexReader(small_index["path"], mode=mode)
        try:
            all_ids = np.arange(N, dtype=np.int64)
            assert np.array_equal(r._read_pq_codes(all_ids), codes), mode
            for u in range(N):
                assert r._read_neighbors(u).tolist() == graph[u][:M], mode
                assert np.array_equal(np.asarray(r._read_full(u)), X[u]), mode
        finally:
            r.close()


def test_eager_and_mmap_return_identical_results(small_index):
    # same algorithm, different storage backing — must agree exactly
    e = dap.IndexReader(small_index["path"], mode="eager")
    m = dap.IndexReader(small_index["path"], mode="mmap")
    try:
        for q in small_index["Q"]:
            ids_e, d_e = e.beam_search(q, k=10, L=64)
            ids_m, d_m = m.beam_search(q, k=10, L=64)
            assert ids_e.tolist() == ids_m.tolist()
            assert np.allclose(d_e, d_m)
    finally:
        e.close()
        m.close()


# ----------------------------------------------------------------------
# measurement tools
# ----------------------------------------------------------------------

def test_exact_knn_matches_naive_bruteforce(small_index):
    X, Q = small_index["X"], small_index["Q"][:5]
    got = dap.exact_knn(X, Q, k=5)
    for i, q in enumerate(Q):
        d = np.einsum("ni,ni->n", X - q, X - q)
        assert got[i].tolist() == np.argsort(d)[:5].tolist()


def test_recall_at_k_definition():
    a = np.array([[1, 2, 3, 4]])
    assert dap.recall_at_k(a, a, k=4) == 1.0
    assert dap.recall_at_k(a, np.array([[5, 6, 7, 8]]), k=4) == 0.0
    assert dap.recall_at_k(a, np.array([[1, 2, 7, 8]]), k=4) == 0.5


# ----------------------------------------------------------------------
# end-to-end quality floor
# ----------------------------------------------------------------------

def test_beam_search_finds_true_neighbours(small_index):
    # deterministic small-scale floor, not a benchmark: with the beam as wide
    # as the dataset and PQ bypassed, the graph must recover essentially the
    # exact top-10; with PQ on, a lower floor guards against gross regressions
    X, Q = small_index["X"], small_index["Q"]
    truth = dap.exact_knn(X, Q, k=10)
    r = dap.IndexReader(small_index["path"], mode="mmap")
    try:
        pred_nopq = np.stack([r.beam_search(q, k=10, L=N, use_pq=False)[0]
                              for q in Q])
        pred_pq = np.stack([r.beam_search(q, k=10, L=128)[0] for q in Q])
    finally:
        r.close()
    assert dap.recall_at_k(pred_nopq, truth, k=10) >= 0.95
    assert dap.recall_at_k(pred_pq, truth, k=10) >= 0.60
