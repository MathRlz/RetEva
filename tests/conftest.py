"""Shared fixtures for the webapi test modules.

`client` is module-scoped: three consumers already used scope="module", and
test_introspection_schema.py's module-scoped `schema` fixture depends on it
(a function-scoped client would be a ScopeMismatch). All imports live inside
the fixture so collection never breaks when fastapi is not installed —
tests that use the fixture skip instead. Modules needing a differently-built
app (test_graph_store.py, test_builder_run.py) shadow this with their own
`client` fixture.
"""

import pytest


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from evaluator.webapi.app import create_app

    return TestClient(create_app())
