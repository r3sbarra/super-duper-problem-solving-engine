"""Shared pytest fixtures.

The vector index is a module-level singleton (``vectorize.vectorization_service``)
backed by a single SQLite connection, which under pytest is ``:memory:``. That
makes it shared by every test in the session: rows one test writes stay visible
to every test that runs after it, so results depend on collection order.

That was harmless while nothing wrote to the index during tests. It stopped
being harmless once the engine gained a write path
(``remember_solution`` / the ``super_solver_remember_solution`` tool), and it
was already a latent hazard for the tests that seed the index themselves. This
fixture resets the index around every test so each one starts empty.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_vector_index():
    """Clear the session-wide vector-index singleton before and after each test.

    Refuses to run against a file-backed database.  The singleton's path is
    resolved at import time, and if that happened outside a test (e.g. during
    collection) it can point at the developer's real index -- deleting from
    that destroys work. ``get_vector_db_path`` now detects pytest by
    ``sys.modules`` as well, so this should never see a real path; the guard is
    here so that if it ever does, the failure is a skip rather than data loss.
    """
    from super_solver.vectorize import vectorization_service

    index = vectorization_service.index
    if getattr(index, "db_path", ":memory:") != ":memory:":
        yield
        return

    conn = index._conn  # one shared in-memory connection

    def _clear():
        with conn:
            conn.execute("DELETE FROM vectors")

    _clear()
    yield
    _clear()
