"""L5: _NodeView isolates the per-branch scope:node attrs. Two concurrent branch views from
the same base must not see each other's pipeline rebinding / current_node, and a top-level
container replacement in one view must not affect the base or a sibling view."""

from evaluator.evaluation.executor.views import _NodeView, _VIEW_LOCAL_ATTRS


class _FakeBase:
    # mimic a RunState with the scope:node fields + a shared (non-node) field
    def __init__(self):
        for a in _VIEW_LOCAL_ATTRS:
            setattr(self, a, f"base_{a}")
        self.shared_field = "shared"


def test_branch_views_isolate_pipeline_rebinding():
    base = _FakeBase()
    a = _NodeView(base, node="nodeA")
    b = _NodeView(base, node="nodeB")
    # each view's current_node is private
    assert a.current_node == "nodeA"
    assert b.current_node == "nodeB"
    # rebinding a node-scoped pipeline in view A doesn't touch base or view B
    pipe_attr = next(iter(_VIEW_LOCAL_ATTRS - {"current_node"}))
    setattr(a, pipe_attr, "A_swapped")
    assert getattr(a, pipe_attr) == "A_swapped"
    assert getattr(b, pipe_attr) == f"base_{pipe_attr}"
    assert getattr(base, pipe_attr) == f"base_{pipe_attr}"


def test_view_delegates_shared_fields_to_base():
    base = _FakeBase()
    v = _NodeView(base, node="n")
    assert v.shared_field == "shared"
    # writing a non-node field goes through to the shared base
    v.shared_field = "changed"
    assert base.shared_field == "changed"
