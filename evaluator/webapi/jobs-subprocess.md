# JobManager — run evaluations in a subprocess (fix webapi segfault)

## Problem
Webapi runs evals in a `ThreadPoolExecutor` worker thread (`jobs.py`). Native ML libs
(PyTorch / OpenMP / HF tokenizers) are not reliably thread-safe off the main thread:
after one model runs, a second model's CPU ops segfault (`SentenceTransformer.encode`
at PHASE 2). CLI runs on the process main thread → fine. Fix: run each job in a **child
process** (spawn) so the work executes on that process's main thread, exactly like the
CLI. Bonus: a segfaulting child no longer takes down the server — it's reported as a
failed job.

## Design
- Parent keeps a `ThreadPoolExecutor` (concurrency limit) but each task is a **supervisor**
  thread that does NO ML work — it only starts/monitors a child process and drains queues.
- `multiprocessing.get_context("spawn")` (never fork — fork-in-thread is the bug).
- Per job: a `progress` Queue and a `result` Queue. Child runs the eval with a
  progress callback that `put()`s events; parent supervisor drains them into
  `push_progress` (unchanged downstream → `_status.html` still polls).
- Child entrypoints are **top-level functions** (picklable); config / test_setups are
  dataclasses/dicts (picklable). No closures cross the process boundary.
- Result: child puts `results.to_dict(include_config=True)` (evaluation) or the matrix
  dict on the result Queue. Exceptions → put `("error", traceback_str)`.
- **Crash detection:** if the child exits with no result and `exitcode != 0` (negative =
  killed by signal, e.g. SIGSEGV), mark the job `failed` with
  `"worker crashed (signal/exit N)"` instead of hanging.
- **Cancel:** `request_cancel` sets the flag; supervisor `proc.terminate()`s a running
  child → status `cancelled`.

## Status: DONE (J1–J3). Subprocess made crashes survivable but did NOT fix the SIGSEGV.

---

# REWRITE: run jobs via the CLI subprocess (drop thread/mp.spawn madness)

The thread→mp.spawn→OMP-pin rabbit hole didn't reliably fix the SIGSEGV. The **CLI
works**. So: each job launches `evaluate.py --config <tmp>.yaml` as a plain OS
subprocess (runs the working bundle path, full isolation). KISS, max robustness,
minimal new code. Decisions: **webapi ingests the leaderboard after the run**;
**phase-level progress by parsing the subprocess stdout**.

## Clean out (jobs.py)
- Remove the mp.spawn machinery: `_run_evaluation_child`, `_run_matrix_child`,
  `_supervise_subprocess`, `_ctx`, `_subprocess` queues, the `_enable_faulthandler`/
  `_pin_torch_threads` child hooks, and the top-of-file `OMP/MKL/...` env block + `# noqa`
  imports. Drop `multiprocessing`/`queue` imports.
- Keep: `JobRecord`, the `JobManager` API (submit/get/list/cancel/push_progress), the
  supervisor `ThreadPoolExecutor` (now supervises a `subprocess.Popen`), and the
  in-process path for injected/custom runners (tests).
- Keep elsewhere (good, unrelated): lazy faiss; env-gated faulthandler in
  `evaluate.py`/`webapi/__main__`; TTS-release in `prepare.py`; service STAGE markers.

## New flow — `_run_cli_subprocess(job_id, config)`
1. Per-job temp dir; `cfg = replace(config, output_dir=<jobdir>)`; `cfg.to_yaml(<jobdir>/config.yaml)`.
2. `Popen([sys.executable, <repo>/evaluate.py, "--config", <cfg.yaml>], cwd=<repo>,
   stdout=PIPE, stderr=STDOUT, text=True, start_new_session=True)`.
   (repo = `Path(evaluator.__file__).parents[1]`.)
3. Stream stdout lines → on a line containing `"PHASE "` / `"STAGE "`, `push_progress`
   (phase = the line, current/total=0). Keep last ~50 lines for error reporting.
4. Exit 0 → read `<jobdir>/<generate_output_filename(cfg)>` (reuse
   `cli.utils.generate_output_filename`) as `job.result`; ingest into the leaderboard at
   `config.output_dir/leaderboard.sqlite` via
   `ExperimentStore.ingest_result(EvaluationResults(metrics=result, config=cfg, metadata=result.get("metadata", {})))`
   — same DB the service used. Non-zero exit → `failed` with the stderr tail (the
   faulthandler dump / traceback lands here; server stays up).
5. Cancel → `os.killpg(os.getpgid(proc.pid), SIGTERM)` (kills DataLoader children too).

## Matrix jobs
`evaluate.py` has no matrix mode. Run each setup as a CLI subprocess (loop), then build
the comparison bundle in the parent (no models → safe). If the comparison builder isn't
cheaply reusable, do eval-jobs first and keep matrix as a follow-up task.

## Tasks
- [ ] C1: rewrite `jobs.py` — rip out mp/OMP; add `_run_cli_subprocess` + helpers
      (write config, parse progress, read+ingest result, cancel via process group).
- [ ] C2: revert the OMP-pin commit's env block / `_pin_torch_threads` (covered by C1).
- [ ] C3: tests — `test_jobs.py` rewritten: in-process path (injected runner) unchanged;
      subprocess path tested with a tiny stub command (fake `evaluate.py` writing a result
      json / exiting nonzero) instead of real models. `pytest tests/test_webapi.py
      tests/test_jobs.py -q` green.
- [ ] C4 (optional): matrix via per-setup CLI subprocess + comparison in parent.

## Verify
- Unit/stub: stubbed subprocess → completed job with result + leaderboard row; nonzero
  exit → failed with stderr tail; cancel kills the process group.
- Live: m4t_asr webapi job → completes (CLI path, no segfault), phase progress shows,
  result + Results-tab populate.

---

# Debugging the service-path SIGSEGV (embedding)

The job child (a clean spawned process, main thread) still SIGSEGVs (exit -11) at the
text-embedding phase → **not** a thread/process issue. Same engine as the CLI
(`evaluate_from_bundle`); the divergence is the **service path**
(`services/_service_run_evaluation` → `ModelServiceProvider` + `load_dataset`/
`build_index` + TTS via `synthesize_missing_query_audio`). The CLI bundle path works.
A native segfault needs the C-level backtrace — `faulthandler`. Decision: debug the
service path; keep the subprocess JobManager.

## D1 — Instrumentation (no model load; implement now)
- [ ] `faulthandler.enable()` at the start of the job child (`_run_evaluation_child` /
      `_run_matrix_child` in `jobs.py`); env-gated `EVALUATOR_FAULTHANDLER=1` in
      `webapi/__main__.py` and `evaluate.py`.
- [ ] Explicit INFO markers in `services/evaluation_service.load_dataset` around the
      TTS-synthesis, corpus-embed, ASR, and query-embed calls so the last log line
      before a crash localizes the stage.

## D2 — Capture backtrace + bisect (user runs; loads models)
- [ ] Rerun the failing webapi job → server log prints the crashing Python+C frame.
- [ ] Standalone repro: `run_evaluation(get_preset("evaluation_config_m4t_asr"))` in a
      plain process with faulthandler → confirms service path crashes outside webapi.
- [ ] Bisection (one var at a time): `audio_synthesis.enabled=false`+real-audio dataset
      (isolate MMS TTS); `service_runtime.startup_mode=eager`; `offload_policy=never`;
      swap labse→bge_m3 / seamless→whisper; CLI bundle path on identical config.

## D3 — Fix (depends on D1/D2 findings)
- [ ] TTS(MMS)+embedding native conflict → synthesize in a separate phase + free the TTS
      model before embedding; or pin the lib. OR load-order → eager-load to match CLI;
      OR OMP/faiss duplicate-runtime → env/import-order. faulthandler frame decides.

Prime suspect: m4t_asr sets `audio_synthesis.enabled: true, provider: mms`, so the
service synthesizes audio for text-only pubmed_qa → loads **MMS VITS** then **labse** in
one process. If the CLI test used real audio (no TTS), that extra native lib is the diff.

## ROOT CAUSE — CONFIRMED (faulthandler backtrace)
Crash in `torch.nn.functional.embedding` (labse BERT) on the first embed batch. The
fault dump's loaded extension modules include **`faiss._swigfaiss_avx512`** — faiss is
imported even though m4t_asr uses `vector_db.type: inmemory`. **faiss + torch both link
OpenMP**; in the webapi child's import order they clash and torch's parallel embedding op
segfaults. CLI's import order happens to be safe → "same config works on CLI".

Source: `storage/vector_store.py:5` is a **module-top `import faiss`**, so any
`import evaluator.storage` (cache, leaderboard — used by webapi/child) pulls faiss in
regardless of backend. (The earlier TTS-release commit was a good hygiene fix but not
the cause.)

## FIX (D3) — lazy faiss import
- [ ] Remove top-level `import faiss` in `storage/vector_store.py`; import faiss **inside**
      `FaissVectorStore`/`FaissGpuVectorStore` `__init__` (+ any method using `faiss.*`).
- [ ] Verify no module-level `faiss.<X>` use; `grep -rn "import faiss" evaluator` → only
      vector_store.py.
- [ ] Result: `inmemory` runs never load faiss → no faiss/torch OpenMP coexistence →
      crash gone; faiss loads only when a Faiss backend is constructed.
- [ ] (Optional, faiss-backed configs only) `faiss.omp_set_num_threads(1)` on Faiss store
      construction.

### Verify
- `python -c "import evaluator.storage, sys; assert 'faiss' not in sys.modules"`.
- FaissVectorStore still works when constructed (faiss imported lazily).
- `pytest tests/test_webapi.py tests/test_jobs.py tests/test_hybrid_retrieval.py -q`.
- User: rerun the m4t_asr webapi job → completes (no SIGSEGV).

## Tasks
### J1 — Subprocess execution core
- Add top-level `_run_evaluation_child(config, progress_q, result_q)` and
  `_run_matrix_child(config, test_setups, baseline_setup_id, progress_q, result_q)` in
  `jobs.py` (import `run_evaluation`/`run_evaluation_matrix` inside the child).
- `JobManager._submit` → submit a `_supervise(job_id, target, args)` task to the thread
  pool. `_supervise` spawns the process, marks running, drains `progress_q`
  (`push_progress`), joins, reads `result_q`.
- `submit_evaluation`/`submit_matrix` pass picklable payloads (config [+ setups]) instead
  of closures. Keep `_evaluation_runner`/`_matrix_runner` injectable for tests (call them
  in-process when a custom runner is provided — see Testing).

### J2 — Cancel + crash handling
- `JobRecord` keeps a handle to the process (or a cancel `Event`); `request_cancel`
  terminates a running child. Map child `exitcode` → failed reason on abnormal exit.

### J3 — Tests + smoke
- Keep `tests/test_webapi.py` job tests green. Tests inject fake runners → must run
  **in-process** (don't spawn): when `evaluation_runner`/`matrix_runner` are non-default,
  `_supervise` calls them directly (so existing mocked tests work unchanged).
- Add a test: a child that exits non-zero → job `failed` with a crash message.
- Live smoke: submit the real `evaluation_config_m4t_asr` job via the UI → completes (no
  segfault); progress shows; result/leaderboard populate.

## Files
- `evaluator/webapi/jobs.py` (core), maybe `evaluator/webapi/app.py` (unchanged wiring).
- `tests/test_webapi.py` (+ a crash test).

## Verification
- `pytest tests/test_webapi.py -q` green.
- Repro that crashed before now succeeds via webapi; a forced child crash → failed job,
  server stays up.
