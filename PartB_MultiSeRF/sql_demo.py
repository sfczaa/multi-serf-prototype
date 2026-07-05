"""
DuckDB UDF demo — the Multi-SeRF index answering a SQL query.

Scope (honest): this registers the Python prototype as a DuckDB scalar UDF so
a filtered nearest-neighbour question can be asked *in SQL*. It demonstrates
the SQL-facing shape of the index, not a DBMS integration — there is no
extension, no persistence, no planner hook, and the timing compares DuckDB's
vectorised native scan against a pure-Python graph walk crossing the UDF
boundary per row.

Run:  py -3 sql_demo.py      (needs `pip install duckdb`; ~15 s)
"""

from __future__ import annotations

import time

import duckdb
import numpy as np
import pandas as pd

import multiserf_proto as ms
from run_experiments_partB import gen_data, gen_queries

N, DIM, K, KNN = 5000, 32, 16, 10
ALPHA = 16.0         # over-fetch comfortably above the recall-0.9 point at narrow B


def main():
    X, a, b = gen_data(N, DIM, seed=0)
    print(f"building Multi-SeRF (K={K}) over n={N}, dim={DIM} ...")
    cs = ms.CompoundSegment(X, a, b, K=K)

    con = duckdb.connect()
    items = pd.DataFrame({"id": np.arange(N, dtype=np.int64),
                          "a": a, "b": b, "vec": list(X)})
    con.execute(f"CREATE TABLE items AS "
                f"SELECT id, a, b, vec::FLOAT[{DIM}] AS vec FROM items")

    def multiserf_knn(qvec, a_lo, a_hi, b_lo, b_hi):
        q = np.asarray(qvec, dtype=np.float32)
        ids, _, _ = cs.query(q, a_lo, a_hi, b_lo, b_hi, KNN, alpha=ALPHA)
        return ids

    con.create_function("multiserf_knn", multiserf_knn,
                        [f"FLOAT[{DIM}]", "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE"],
                        "BIGINT[]")

    a_lo, a_hi, b_lo, b_hi = 0.2, 0.9, 0.50, 0.51   # narrow B window (~1%)
    Q = gen_queries(10, DIM, seed=12345)

    sql_native = f"""
        SELECT id FROM items
        WHERE a BETWEEN ? AND ? AND b BETWEEN ? AND ?
        ORDER BY array_distance(vec, ?::FLOAT[{DIM}]) LIMIT {KNN}"""
    sql_udf = f"""
        SELECT unnest(multiserf_knn(?::FLOAT[{DIM}], ?, ?, ?, ?)) AS id"""

    print(f"query: {KNN}-NN with a in [{a_lo},{a_hi}], b in [{b_lo},{b_hi}] "
          f"(B window ~1% selective)\n")
    recalls, t_nat, t_udf = [], 0.0, 0.0
    for i in range(Q.shape[0]):
        qv = Q[i].tolist()
        t0 = time.perf_counter()
        exact = [r[0] for r in con.execute(
            sql_native, [a_lo, a_hi, b_lo, b_hi, qv]).fetchall()]
        t_nat += time.perf_counter() - t0
        t0 = time.perf_counter()
        approx = [r[0] for r in con.execute(
            sql_udf, [qv, a_lo, a_hi, b_lo, b_hi]).fetchall()]
        t_udf += time.perf_counter() - t0
        recalls.append(ms.recall_at_k(approx, exact, KNN))

    nq = Q.shape[0]
    print("both statements answer the same question in SQL:")
    print(f"  native scan  : ORDER BY array_distance(...) LIMIT {KNN}   "
          f"exact, {1e3 * t_nat / nq:6.1f} ms/query")
    print(f"  index UDF    : unnest(multiserf_knn(...))            "
          f"recall {float(np.mean(recalls)):.2f} vs exact, "
          f"{1e3 * t_udf / nq:6.1f} ms/query")
    print(f"\nexample rows (last query): {approx[:5]} ...")
    print("\ncaveats: scalar-UDF demo only - no extension/persistence/planner")
    print("integration. This demo shows the SQL-facing API shape and result")
    print("agreement; the toy wall-clock numbers above are not a performance")
    print("claim in either direction (see results.md for the measured claims).")


if __name__ == "__main__":
    main()
