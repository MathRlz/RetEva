"""Runtime environment diagnostics.

Surfaces the Python interpreter and the loaded native threading stack so env-specific
native crashes are diagnosable from the logs. The webapi runs evals in a subprocess using
``sys.executable``; if that interpreter's environment differs from the one where the CLI
works (e.g. a container ``/opt/venv`` vs a conda env), or it loads two OpenMP runtimes
(torch's ``libgomp`` + MKL's ``libiomp`` via MKL-linked numpy/scipy/sklearn), a torch op
can SIGSEGV. Logging this once at run start makes both causes visible.
"""
from __future__ import annotations

import sys
from typing import List


def _versions() -> List[str]:
    out: List[str] = []
    for mod in ("torch", "numpy", "transformers", "sentence_transformers"):
        try:
            out.append(f"{mod}={getattr(__import__(mod), '__version__', '?')}")
        except Exception:
            out.append(f"{mod}=<unavailable>")
    return out


def _openmp_runtimes() -> str:
    """List loaded OpenMP/BLAS runtimes; more than one OpenMP entry = duplicate runtime."""
    try:
        import threadpoolctl
        parts = [
            f"{i.get('user_api', '?')}:{i.get('prefix', '?')}({i.get('num_threads', '?')})"
            for i in threadpoolctl.threadpool_info()
        ]
        return ", ".join(parts) if parts else "none"
    except Exception:
        return "threadpoolctl-unavailable"


def _process_state() -> str:
    """Process state that survives ``exec`` and varies by launch context.

    A webapi job is fork+exec'd from a torch-loaded, multi-threaded asyncio parent; the CLI
    is exec'd from a clean shell. The inherited **signal mask** and **CPU affinity** are the
    exec-surviving differences most likely to segfault a native torch op — log them so the
    two launches can be diffed.
    """
    import os
    parts = [f"cpu_count={os.cpu_count()}"]
    try:
        parts.append(f"affinity={len(os.sched_getaffinity(0))}")
    except (AttributeError, OSError):
        parts.append("affinity=?")
    try:
        import torch
        parts.append(f"torch_threads={torch.get_num_threads()}/{torch.get_num_interop_threads()}")
        if torch.cuda.is_available():
            # Names matter: a heterogeneous box (e.g. a good dGPU + an unsupported iGPU)
            # will crash any op placed on the bad device. Surfacing them makes that obvious.
            gpus = [f"{i}:{torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
            parts.append("gpus=[" + "; ".join(gpus) + "]")
    except Exception:
        pass
    try:
        import signal
        blocked = sorted(int(s) for s in signal.pthread_sigmask(signal.SIG_BLOCK, []))
        parts.append(f"sigmask_blocked={blocked}")
    except (AttributeError, ValueError, OSError):
        pass
    try:
        import resource
        parts.append(f"rlimit_stack={resource.getrlimit(resource.RLIMIT_STACK)[0]}")
    except Exception:
        pass
    env = {k: os.environ.get(k) for k in
           ("OMP_NUM_THREADS", "MKL_THREADING_LAYER", "OMP_PROC_BIND", "GOMP_CPU_AFFINITY")}
    parts.append("env=" + (",".join(f"{k}={v}" for k, v in env.items() if v is not None) or "-"))
    return " ".join(parts)


def runtime_summary() -> str:
    """One-line summary: interpreter, library versions, threading runtimes, process state."""
    return (
        f"python={sys.version.split()[0]} exe={sys.executable} | "
        + " | ".join(_versions())
        + f" | openmp=[{_openmp_runtimes()}]"
        + f" | proc=[{_process_state()}]"
    )


def log_runtime_summary(logger) -> None:
    """Best-effort log of :func:`runtime_summary` (never raises)."""
    try:
        logger.info("RUNTIME %s", runtime_summary())
    except Exception:
        pass
