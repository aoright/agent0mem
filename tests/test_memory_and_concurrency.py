import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import threading
import time
import pytest
import numpy as np

from app.core.clean_search import (
    _get_user_build_lock,
    _INDEX_CACHE,
    _INDEX_LOCK,
    _load_vectors_matrix,
    _get_index,
    _evict_cache_if_needed,
    _INDEX_CACHE_MAX_ROWS
)
from app.core.embeddings import encode_vector
from app.database import init_db, get_connection


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_vector_matrix_loading_directly_to_numpy():
    conn = get_connection()
    c = conn.cursor()
    uid = "test_user_vec_mat"
    c.execute("DELETE FROM memories WHERE user_id = ?", (uid,))
    
    # Insert dummy memories and float16 embeddings
    mids = [f"mem_{uid}_{i}" for i in range(10)]
    for i, mid in enumerate(mids):
        c.execute(
            "INSERT INTO memories (memory_id, user_id, session_id, role, content, item_type) VALUES (?, ?, ?, ?, ?, ?)",
            (mid, uid, "sess1", "user", f"Content number {i}", "raw")
        )
        vec = [0.1 * (i + 1)] * 128
        raw_blob = encode_vector(vec)
        c.execute("INSERT OR REPLACE INTO embeddings (memory_id, vector) VALUES (?, ?)", (mid, raw_blob))
    conn.commit()

    vec_ids, mat, vecs = _load_vectors_matrix(mids)
    assert len(vec_ids) == 10
    assert mat is not None
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (10, 128)
    assert vecs is None  # vecs should NOT be allocated when numpy is available!
    
    # Check normalization
    norms = np.linalg.norm(mat, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_concurrent_index_build_single_flight(monkeypatch):
    """Verify that multiple threads requesting index for the same user only trigger 1 build."""
    uid = "test_user_concurrent_stampede"
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM memories WHERE user_id = ?", (uid,))
    c.execute(
        "INSERT INTO memories (memory_id, user_id, session_id, role, content, item_type) VALUES (?, ?, ?, ?, ?, ?)",
        (f"mem_{uid}_1", uid, "sess1", "user", "Hello world", "raw")
    )
    conn.commit()

    build_count = 0
    import app.core.clean_search as cs_mod
    original_build = cs_mod._build_index

    def wrapped_build(u):
        nonlocal build_count
        build_count += 1
        time.sleep(0.05)  # Simulate slow DB load
        return original_build(u)

    monkeypatch.setattr(cs_mod, "_build_index", wrapped_build)

    with _INDEX_LOCK:
        _INDEX_CACHE.pop(uid, None)

    threads = []
    results = []

    def worker():
        idx = _get_index(uid)
        results.append(idx)

    for _ in range(8):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(results) == 8
    # All 8 threads must have gotten the index, but build should only execute ONCE!
    assert build_count == 1


def test_cache_row_eviction():
    with _INDEX_LOCK:
        _INDEX_CACHE.clear()
        # Mock 2 large user index entries
        dummy_rows_1 = [{"memory_id": f"m1_{i}"} for i in range(30000)]
        dummy_rows_2 = [{"memory_id": f"m2_{i}"} for i in range(30000)]
        
        idx1 = (dummy_rows_1, {}, None, [], None, None, {}, [], {})
        idx2 = (dummy_rows_2, {}, None, [], None, None, {}, [], {})
        
        _INDEX_CACHE["u1"] = (("k1",), idx1)
        _INDEX_CACHE["u2"] = (("k2",), idx2)
        
        # Total rows is 60,000 > _INDEX_CACHE_MAX_ROWS (50,000)
        _evict_cache_if_needed()
        
        # Oldest user u1 should have been evicted!
        assert "u1" not in _INDEX_CACHE
        assert "u2" in _INDEX_CACHE
