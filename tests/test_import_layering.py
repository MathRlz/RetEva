"""Package layering: module-level imports form a DAG that respects LAYERS; no cross-package
import of an underscore name; function-local imports of internal modules only where allowed."""
import ast
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent / "evaluator"

LAYERS = [
    {"constants", "errors", "logging_config", "utils"},
    {"core", "config"},
    {"metrics", "llm_client", "storage", "devices", "tracking"},
    {"models"},
    {"datasets"},
    {"pipeline"},
    {"evaluation", "judge", "analysis", "benchmarks"},
    {"services"},
    {"cli", "webapi", "api", "public_api", "diagnostics"},
]
RANK = {pkg: i for i, layer in enumerate(LAYERS) for pkg in layer}

# module → reason. Function-local imports of evaluator internals are allowed only here.
LAZY_ALLOWED = {
    "config/evaluation.py": "EvaluationConfig.graph builds via pipeline (documented downward edge); "
                            "device-pool allocation in with_auto_devices (config → devices)",
    "config/graph_config.py": "node-centric translation needs the graph vocabulary (documented downward edge, spec P3)",
    "config/node_config.py": "node-centric translation needs the graph vocabulary (documented downward edge, spec P3)",
    "config/serialization.py": "node-centric translation needs the graph vocabulary (documented downward edge, spec P3)",
    "config/validation.py": "node-centric translation needs the graph vocabulary (documented downward edge, spec P3)",
    "config/dataset_descriptors.py": "descriptor field validation needs the artifact registry",
    "config/model_presets.py": "cycle model_presets -> evaluation -> loading -> model_presets "
                               "(a preset is 'a file that parses as EvaluationConfig')",
    "storage/registry.py": "importing the chromadb / qdrant wrappers pulls their (installed) "
                           "package; the factories defer that cost to the run that asks for it",
    "models/factory.py": "the a2e builders read Params off a torch/transformers-heavy model "
                         "module; loading those eagerly would defeat the lazy model exports",
    "pipeline/factory.py": "the clap_style builder names a torch/transformers-heavy model class",
    "pipeline/graph/operators.py": "cycle operators -> registry -> operators: the alias "
                                   "bijection asks the node registry which operators exist",
}


def _pkg(rel):
    parts = rel.split("/")
    return parts[0].removesuffix(".py") if len(parts) == 1 else parts[0]


def _imports(path, rel):
    try:
        tree = ast.parse(path.read_text())
    except Exception as e:
        raise ValueError(f"Failed to parse {rel}: {e}") from e
    parts = rel.split("/")
    top, local, private = [], [], []

    def record(node, deferred):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = parts[:-1][: len(parts) - 1 - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
                target = mod.split(".")[0] if mod else _pkg(rel)
            elif node.module == "evaluator":
                target = "evaluator"
            elif node.module and node.module.startswith("evaluator."):
                target = node.module.split(".")[1]
            else:
                return
            names = [a.name for a in node.names]
            (local if deferred else top).append((target, node.lineno, names))
            if target != _pkg(rel):
                private.extend(
                    (n, node.lineno) for n in names if n.startswith("_") and n != "__version__"
                )
        else:
            for a in node.names:
                if a.name == "evaluator":
                    (local if deferred else top).append(("evaluator", node.lineno, [a.name]))
                elif a.name.startswith("evaluator."):
                    (local if deferred else top).append((a.name.split(".")[1], node.lineno, [a.name]))

    # An import is deferred only when it sits inside a function body. One nested in a
    # module-level ``if TYPE_CHECKING:`` or ``try:`` guard runs at import time and is a
    # module-level import for every purpose here, whatever its column.
    def visit(node, deferred):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                record(child, deferred)
            visit(child, deferred or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)))

    visit(tree, False)
    return top, local, private


def _modules():
    for p in sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if "__pycache__" in rel:
            continue
        yield p, rel


def test_module_level_imports_respect_layer_order():
    bad = []
    for p, rel in _modules():
        try:
            src = _pkg(rel)
            top, _, _ = _imports(p, rel)
            for target, line, names in top:
                if target == src or target not in RANK or src not in RANK:
                    continue
                if RANK[target] > RANK[src]:
                    bad.append(f"{rel}:{line} imports {target} ({names}) from a higher layer")
        except Exception as e:
            bad.append(f"{rel}: {e}")
    assert not bad, "\n".join(bad)


def test_module_level_package_graph_is_acyclic():
    """Layer order alone leaves same-rank packages free to import each other. Build the
    package-level graph from every module-level import and prove it is a DAG — a cycle is an
    import-time deadlock waiting for whichever module is imported first."""
    edges = {}
    for p, rel in _modules():
        src = _pkg(rel)
        top, _, _ = _imports(p, rel)
        edges.setdefault(src, set()).update(
            target for target, _, _ in top if target != src
        )

    WHITE, GREY, BLACK = 0, 1, 2
    color = {}
    cycle = []

    def walk(node, stack):
        if color.get(node, WHITE) == BLACK or cycle:
            return
        if color.get(node, WHITE) == GREY:
            cycle.extend(stack[stack.index(node):] + [node])
            return
        color[node] = GREY
        for nxt in sorted(edges.get(node, ())):
            walk(nxt, stack + [nxt])
            if cycle:
                return
        color[node] = BLACK

    for pkg in sorted(edges):
        walk(pkg, [pkg])
        if cycle:
            break

    assert not cycle, "package-level import cycle: " + " -> ".join(cycle)


def test_no_cross_package_private_imports():
    bad = []
    for p, rel in _modules():
        try:
            _, _, private = _imports(p, rel)
            bad += [f"{rel}:{line} imports private {n}" for n, line in private]
        except Exception as e:
            bad.append(f"{rel}: {e}")
    assert not bad, "\n".join(bad)


def test_function_local_internal_imports_are_allowlisted():
    bad = []
    for p, rel in _modules():
        try:
            _, local, _ = _imports(p, rel)
            if local and rel not in LAZY_ALLOWED:
                bad.append(f"{rel}: {len(local)} function-local internal imports (first at line {local[0][1]})")
        except Exception as e:
            bad.append(f"{rel}: {e}")
    assert not bad, "\n".join(bad)
