import pytest
from app.database import search_hybrid, init_db, get_connection

@pytest.fixture(autouse=True)
def setup_test_db():
    init_db()

def test_django_cross_module_generalization():
    """Verify that a Django query targeting models demotes competing distractor patches from contrib.auth."""
    conn = get_connection()
    c = conn.cursor()
    uid = "test_user_django_generalization"
    c.execute("DELETE FROM memories WHERE user_id = ?", (uid,))
    
    # Ingest a target patch in django.db.models
    c.execute("""
        INSERT INTO memories (memory_id, user_id, session_id, role, content, item_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        "mem_django_models", uid, "sess_models", "assistant",
        "Fix for django models field validation error in django/db/models/fields/__init__.py:\n"
        "diff --git a/django/db/models/fields/__init__.py b/django/db/models/fields/__init__.py\n"
        "--- a/django/db/models/fields/__init__.py\n"
        "+++ b/django/db/models/fields/__init__.py\n"
        "@@ -100,5 +100,7 @@ class Field:\n"
        "+    def validate_field(self):\n"
        "+        return True\n",
        "raw_message"
    ))

    # Ingest a distractor patch in django.contrib.auth
    c.execute("""
        INSERT INTO memories (memory_id, user_id, session_id, role, content, item_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        "mem_django_auth", uid, "sess_auth", "assistant",
        "Fix for django user permission caching in django/contrib/auth/models.py:\n"
        "diff --git a/django/contrib/auth/models.py b/django/contrib/auth/models.py\n"
        "--- a/django/contrib/auth/models.py\n"
        "+++ b/django/contrib/auth/models.py\n"
        "@@ -50,5 +50,7 @@ class Permission:\n"
        "+    def clear_cache(self):\n"
        "+        pass\n",
        "raw_message"
    ))
    conn.commit()

    query = "django.db.models field validation error in fields/__init__.py"
    results = search_hybrid(uid, query, top_k=3)
    assert len(results) > 0
    assert "django/db/models/fields" in results[0]["content"]

def test_sympy_cross_module_generalization():
    """Verify that a SymPy query targeting solvers demotes competing distractor patches from polys."""
    conn = get_connection()
    c = conn.cursor()
    uid = "test_user_sympy_generalization"
    c.execute("DELETE FROM memories WHERE user_id = ?", (uid,))

    c.execute("""
        INSERT INTO memories (memory_id, user_id, session_id, role, content, item_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        "mem_sympy_solvers", uid, "sess_solvers", "assistant",
        "Fix for sympy differential equation solver in sympy/solvers/ode.py:\n"
        "diff --git a/sympy/solvers/ode.py b/sympy/solvers/ode.py\n"
        "--- a/sympy/solvers/ode.py\n"
        "+++ b/sympy/solvers/ode.py\n"
        "@@ -200,5 +200,7 @@ def dsolve(eq):\n"
        "+    return solution\n",
        "raw_message"
    ))

    c.execute("""
        INSERT INTO memories (memory_id, user_id, session_id, role, content, item_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        "mem_sympy_polys", uid, "sess_polys", "assistant",
        "Fix for sympy polynomial factorization in sympy/polys/polytools.py:\n"
        "diff --git a/sympy/polys/polytools.py b/sympy/polys/polytools.py\n"
        "--- a/sympy/polys/polytools.py\n"
        "+++ b/sympy/polys/polytools.py\n"
        "@@ -300,5 +300,7 @@ def factor(f):\n"
        "+    return factors\n",
        "raw_message"
    ))
    conn.commit()

    query = "sympy.solvers dsolve differential equation ode.py"
    results = search_hybrid(uid, query, top_k=3)
    assert len(results) > 0
    assert "sympy/solvers/ode.py" in results[0]["content"]

def test_astropy_cross_module_generalization():
    """Verify that an Astropy timeseries query continues to cleanly isolate timeseries from table."""
    conn = get_connection()
    c = conn.cursor()
    uid = "test_user_astropy_generalization"
    c.execute("DELETE FROM memories WHERE user_id = ?", (uid,))

    c.execute("""
        INSERT INTO memories (memory_id, user_id, session_id, role, content, item_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        "mem_astropy_table", uid, "sess_table", "assistant",
        "Fix for astropy table index removal:\n"
        "diff --git a/astropy/table/table.py b/astropy/table/table.py\n"
        "--- a/astropy/table/table.py\n"
        "+++ b/astropy/table/table.py\n"
        "@@ -100,5 +100,7 @@ class Table:\n"
        "+    def remove_indices(self):\n"
        "+        pass\n",
        "raw_message"
    ))

    c.execute("""
        INSERT INTO memories (memory_id, user_id, session_id, role, content, item_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        "mem_astropy_timeseries", uid, "sess_timeseries", "assistant",
        "Architectural definition for astropy timeseries core.py:\n"
        "TimeSeries subclasses QTable and requires 'time' as the first column.\n"
        "_check_required_columns verifies required columns and raises ValueError with clear message.",
        "proposition"
    ))
    conn.commit()

    query = "astropy timeseries core.py _check_required_columns missing required columns ValueError"
    results = search_hybrid(uid, query, top_k=3)
    assert len(results) > 0
    assert "astropy/table/table.py" not in results[0]["content"]
    assert "TimeSeries" in results[0]["content"] or "core.py" in results[0]["content"]
