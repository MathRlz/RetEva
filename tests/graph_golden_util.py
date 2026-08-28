"""Shared helpers for the graph-parity golden locks.

`graph_signature(config_path)` — order-INsensitive signature of the preview graph
(`build_graph_for_config`): per node (stage, kind, sorted deps, sorted params, sorted bindings).
Frozen in tests/data/graph_golden.json.

`run_order_signature(config_path)` — order-SENSITIVE signature of the RUN graph
(`build_run_graph` shape, `attach_fields=False`): per node the UNSORTED binding tuple +
`input_aliases`. Frozen in tests/data/graph_order_golden.json. This exists because the runtime
read is order-sensitive (`reversed(producers)` newest-first + OneOf candidate priority) while
the plain golden sorts — the explicit-edges migration (E1, docs/archive/ARCHITECTURE_TASKS.md) must
reproduce binding ORDER, not just the binding set.
"""
import yaml

from evaluator.config.evaluation import EvaluationConfig
from evaluator.config.graph_config import build_evaluation_config_kwargs
from evaluator.pipeline.graph.modes import build_graph_for_config, build_run_graph
from evaluator.pipeline.graph.operators import node_kind


def _load(config_path):
    raw = yaml.safe_load(open(config_path))
    return EvaluationConfig.from_dict(build_evaluation_config_kwargs(raw), validate=False)


def graph_signature(config_path):
    g = build_graph_for_config(_load(config_path))
    sig = {}
    for n in g.nodes:
        sig[n.id] = [
            n.stage,
            node_kind(n.stage, n.params),
            sorted(n.depends_on),
            sorted([k, repr(v)] for k, v in (n.params or {}).items()),
            sorted([p, s] for p, s in n.bindings),
        ]
    return sig


def build_run_shape(cfg):
    """The RUN-path graph for an explicit-node config (attach_fields=False). Pipelines are
    never touched on the explicit-nodes path (feature derivation is skipped), so None is fine."""
    return build_run_graph(
        cfg.graph_template,
        graph_override=cfg.graph_override,
        embedding_fusion_config=None,
        query_opt_config=None,
        retrieval_pipeline=None,
        eval_config=cfg,
        trace_limit=0,
    )


def run_order_signature(config_path):
    g = build_run_shape(_load(config_path))
    sig = {}
    for n in g.nodes:
        sig[n.id] = [
            n.stage,
            [list(b) for b in n.bindings],                       # UNSORTED — order is semantics
            [[k, list(c)] for k, c in (n.input_aliases or ())],  # OneOf candidate priority
            sorted(n.depends_on),
        ]
    return sig
