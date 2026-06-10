# Results Tab — Plan

Server-rendered Jinja2 + htmx + one `app.css` (no SPA). Reuse the existing leaderboard
route/partials. Decisions: "delete cache as well" deletes the run's **vector-DB cache**
(its key is already stored in run metadata); the test-polluted `cache_meta` rows get
purged and the test fixed.

## R1 — Fix `cache_meta` pollution (item: "so many cache_meta results") ✅
- [x] Root cause: `tests/test_api.py` ran an eval named `cache_meta` against the
      default `output_dir`, ingesting into the real `evaluation_results/leaderboard.sqlite`.
- [x] Point that test's `config.output_dir` at a tempdir (sibling tests checked — only leak).
- [x] Purge existing `cache_meta` rows from `evaluation_results/leaderboard.sqlite`.

## R2 — Auto-read metric list (item: "Fix choosing metric") ✅
- [x] `ExperimentStore.available_metrics()` — sorted union of numeric metric keys.
- [x] `results.html`: metric `<select>` populated from it (fallback MRR); `ui_results`
      passes the list.

## R3 — Fuzzy name search + category filters (items 1–2) ✅
- [x] `query_leaderboard`/`leaderboard_rows` gained `name` (LIKE on experiment_name) +
      model filters (`asr_model_type`/`text_emb_model_type`/`audio_emb_model_type`, read
      from `config_json.model`), plus existing dataset/mode filters.
- [x] `ExperimentStore.filter_options()` — distinct datasets/modes/model types.
- [x] `results.html`: search box + metric/dataset/pipeline/model `<select>`s in
      `#lb-controls`, composed via `hx-include`, swapping into `#leaderboard`.

## R4 — Delete record + confirm + delete-cache (item 3) ✅
- [x] `ExperimentStore.delete_run(run_id)`.
- [x] `CacheManager.delete_vector_db(db_key)` — unlink `<key>_vectors.npy` +
      `<key>_metadata.json` and remove the manifest row (+ its artifact).
- [x] `DELETE /api/leaderboard/runs/{run_id}` + shared `delete_run_and_cache()` helper;
      reads `metadata.cache.load.vector_cache_key` and deletes via
      `CacheManager(cache_dir=run.config.cache.cache_dir)`.
- [x] UI: per-row ✕ button → `_delete_confirm.html` (with **Delete cache as well**
      checkbox) → `POST /ui/runs/{id}/delete` (preserves filters via `#lb-controls`) →
      re-renders `#leaderboard`.

## Verification
- Unit: available_metrics/filter_options/delete_run on a temp DB; delete_vector_db
  removes artifacts + manifest rows; `pytest tests/test_webapi.py tests/test_api.py
  tests/test_leaderboard_store.py tests/test_cache*.py -q` green (test_api stops writing
  to the real DB).
- Live (`evaluator-webapi --port 8078`): metric dropdown auto-populated; search +
  filters narrow rows; Delete → confirm → row gone; "delete cache" removes the run's
  vector-DB cache; `cache_meta` rows gone from the real DB.
