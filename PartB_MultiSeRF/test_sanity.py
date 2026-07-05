"""
Sanity tests for the Multi-SeRF prototype.

These do not re-verify the experiment numbers; they pin the two properties the
experiment's validity rests on:

  1. `recall_at_k` measures against the achievable truth set, so queries whose
     predicates admit fewer than k points are not unfairly penalised.
  2. `CompoundSegment` with K=1 is *exactly* the SeRF+ResidualB baseline path —
     the same SegmentGraph1D search, no bucket effects. The headline ratio
     isolates B-bucketing only if this holds.

Run:  py -3 -m pytest test_sanity.py -q     (fast: n=300, dim=8)
"""

from __future__ import annotations

import math

import numpy as np

import multiserf_proto as ms


# ----------------------------------------------------------------------
# recall_at_k
# ----------------------------------------------------------------------

def test_recall_perfect_and_partial():
    assert ms.recall_at_k([1, 2, 3], [1, 2, 3], k=3) == 1.0
    assert ms.recall_at_k([1, 9, 8], [1, 2, 3], k=3) == 1.0 / 3.0
    assert ms.recall_at_k([], [1, 2, 3], k=3) == 0.0


def test_recall_empty_truth_is_trivially_correct():
    # if no point satisfies the predicates, returning nothing is right
    assert ms.recall_at_k([], [], k=10) == 1.0
    assert ms.recall_at_k([5, 6], [], k=10) == 1.0


def test_recall_denominator_is_truth_size_when_smaller_than_k():
    # predicate admits only 2 points; recovering both must count as recall 1.0,
    # not 2/k — otherwise narrow-range queries could never reach the 0.9 floor
    assert ms.recall_at_k([1, 2], [1, 2], k=10) == 1.0
    assert ms.recall_at_k([1, 7], [1, 2], k=10) == 0.5


def test_recall_only_counts_topk_predictions():
    # a hit beyond position k must not count
    assert ms.recall_at_k([9, 8, 1], [1, 2], k=2) == 0.0


# ----------------------------------------------------------------------
# CompoundSegment
# ----------------------------------------------------------------------

def _small_dataset(n=300, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    a = rng.random(n)
    b = rng.random(n)
    return X, a, b


def test_k1_is_exactly_the_baseline_graph_path():
    # the experiment's headline ratio isolates B-bucketing only because both
    # arms share the same code: K=1 must reduce to one full-range SegmentGraph1D
    X, a, b = _small_dataset()
    cs1 = ms.CompoundSegment(X, a, b, K=1, M=8, ef=32)
    assert len(cs1.buckets) == 1
    sg = cs1.buckets[0].sg

    rng = np.random.default_rng(7)
    k, alpha = 10, 2.0
    kp = max(k, int(math.ceil(alpha * k)))
    ef_q = max(kp, 32)
    for _ in range(5):
        q = rng.standard_normal(X.shape[1]).astype(np.float32)
        ids_cs, ds_cs, nb = cs1.query(q, 0.2, 0.9, 0.0, 1.0, k, alpha=alpha)
        ids_sg, ds_sg = sg.query(q, 0.2, 0.9, kp, ef_q)
        assert nb == 1
        assert ids_cs == ids_sg[:k]
        assert ds_cs == ds_sg[:k]


def test_query_respects_both_range_predicates():
    # bucket routing must not leak points outside [a_lo,a_hi] x [b_lo,b_hi]:
    # boundary buckets are only partially inside the B range and must be
    # residual-filtered on b
    X, a, b = _small_dataset()
    cs = ms.CompoundSegment(X, a, b, K=8, M=8, ef=32)
    rng = np.random.default_rng(11)
    for _ in range(10):
        q = rng.standard_normal(X.shape[1]).astype(np.float32)
        a_lo, a_hi = 0.1, 0.8
        b_lo, b_hi = 0.3, 0.55
        ids, _, _ = cs.query(q, a_lo, a_hi, b_lo, b_hi, k=10, alpha=4.0)
        for g in ids:
            assert a_lo <= a[g] <= a_hi
            assert b_lo <= b[g] <= b_hi


def test_adaptive_estimate_and_routing():
    # the adaptive index must (a) estimate s_B in the right decade from bucket
    # metadata alone and (b) return byte-identical results to whichever arm it
    # routes to — it adds routing, never a third behaviour
    X, a, b = _small_dataset()
    cs1 = ms.CompoundSegment(X, a, b, K=1, M=8, ef=32)
    csK = ms.CompoundSegment(X, a, b, K=8, M=8, ef=32)
    ai = ms.AdaptiveIndex(cs1, csK, tau=0.15)

    assert abs(ai.estimate_sb(0.0, 1.0) - 1.0) < 1e-9
    assert 0.02 <= ai.estimate_sb(0.40, 0.45) <= 0.10   # ~5% window

    q = np.random.default_rng(5).standard_normal(X.shape[1]).astype(np.float32)
    ids, ds, vb, used = ai.query(q, 0.0, 1.0, 0.40, 0.45, k=5, alpha=4.0, ef=64)
    ids2, ds2, vb2 = csK.query(q, 0.0, 1.0, 0.40, 0.45, 5, alpha=4.0, ef=64)
    assert used is True
    assert (ids, ds, vb) == (ids2, ds2, vb2)

    ids, ds, vb, used = ai.query(q, 0.0, 1.0, 0.05, 0.95, k=5, alpha=4.0, ef=64)
    ids2, ds2, vb2 = cs1.query(q, 0.0, 1.0, 0.05, 0.95, 5, alpha=4.0, ef=64)
    assert used is False
    assert (ids, ds, vb) == (ids2, ds2, vb2)


def test_gen_data_bcorr_zero_reproduces_recorded_stream():
    # every recorded results_partB_*.json depends on this exact RNG stream;
    # the --b-corr feature must not perturb the default path
    from run_experiments_partB import gen_data
    rng = np.random.default_rng(3)
    X0 = rng.standard_normal((200, 4)).astype(np.float32)
    a0 = rng.random(200)
    b0 = rng.random(200)
    X, a, b = gen_data(200, 4, seed=3, b_corr=0.0)
    assert np.array_equal(X, X0)
    assert np.array_equal(a, a0)
    assert np.array_equal(b, b0)


def test_gen_data_bcorr_changes_coupling_not_marginal():
    # equal-frequency bucketing must see the same Uniform[0,1] marginal; only
    # the coupling between B and the vectors changes
    from run_experiments_partB import gen_data
    X, a, b = gen_data(2000, 8, seed=0, b_corr=0.8)
    Xi, ai, _ = gen_data(2000, 8, seed=0, b_corr=0.0)
    assert np.array_equal(X, Xi)
    assert np.array_equal(a, ai)
    # rank transform makes the marginal exactly uniform on (0, 1)
    assert np.allclose(np.sort(b), (np.arange(2000) + 0.5) / 2000)
    # and B strongly rank-correlated with the first vector coordinate
    rb = np.argsort(np.argsort(b))
    rx = np.argsort(np.argsort(X[:, 0]))
    assert np.corrcoef(rb, rx)[0, 1] > 0.7


def test_bucketed_search_still_finds_the_true_neighbours():
    # end-to-end: on a small deterministic dataset, K=8 with generous
    # over-fetch must recover essentially the exact filtered top-k
    X, a, b = _small_dataset()
    cs = ms.CompoundSegment(X, a, b, K=8, M=8, ef=32)
    rng = np.random.default_rng(23)
    recalls = []
    for _ in range(20):
        q = rng.standard_normal(X.shape[1]).astype(np.float32)
        truth = ms.exact_filtered_knn(X, a, b, q, 0.0, 1.0, 0.2, 0.6, k=10)
        ids, _, _ = cs.query(q, 0.0, 1.0, 0.2, 0.6, k=10, alpha=8.0)
        recalls.append(ms.recall_at_k(ids, truth, k=10))
    assert float(np.mean(recalls)) >= 0.9
