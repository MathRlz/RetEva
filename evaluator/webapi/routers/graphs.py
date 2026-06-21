"""Saved builder graphs — persist a named canvas spec and reopen it (P1).

The builder can save the graph it's drawing and reopen one from a gallery. A saved graph is the
canvas export; ``GET`` returns it plus a render payload so the canvas can redraw it directly.
"""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from evaluator.webapi.form_builder import spec_to_render_payload
from evaluator.webapi.graph_store import GraphStore


def build_graphs_router(store: GraphStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/graphs", summary="List saved builder graphs")
    def list_graphs() -> Dict[str, Any]:
        return {"graphs": store.list()}

    @router.post("/api/graphs", summary="Save a builder graph by name")
    def save_graph(payload: Dict[str, Any]) -> Dict[str, str]:
        name = str(payload.get("name") or "").strip()
        spec = payload.get("spec")
        if not name or not isinstance(spec, dict):
            raise HTTPException(status_code=400, detail="Provide a 'name' and a graph 'spec'.")
        try:
            return {"name": store.save(name, spec)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/graphs/{name}", summary="Load a saved graph (+ render payload)")
    def get_graph(name: str) -> Dict[str, Any]:
        try:
            spec = store.get(name)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Graph not found") from exc
        try:
            render = spec_to_render_payload(spec)
        except Exception as exc:  # noqa: BLE001 — a saved graph that no longer builds → 400
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"name": name, "spec": spec, "render": render}

    @router.delete("/api/graphs/{name}", summary="Delete a saved graph")
    def delete_graph(name: str) -> Dict[str, str]:
        try:
            store.delete(name)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Graph not found") from exc
        return {"deleted": name}

    return router
