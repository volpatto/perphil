"""
PETSc performance profiling for DPP experiments.

Backends (in order when backend="auto"):
1) JSON (-log_view :tmp.json:json + PETSc.Log.view) → parse, average across ranks.
2) ASCII (Viewer ASCII, one file per rank) → parse, average across ranks.
3) Events API (PETSc.Log.Event.getPerfInfo before/after) → robust times & FLOPS.
4) Stage API (per-stage before/after snapshot).
5) Wallclock fallback (total only under KSPSolve).

Records:
- Event times (and FLOPS) per logical event, plus flops_total.
- Memory metrics: RSS peak/delta per rank, Mat/PMat/Factor nz and memory (MB).
- Universal time metrics: time_total (avg wall per run) and time_total_repeats (sum across repeats).

Optionally applies manufactured pressures as Dirichlet BCs (use_manufactured=True).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import json
import os
import re
import tempfile
import time
import resource  # RSS metrics

import numpy as np
import pandas as pd

import firedrake as fd
from firedrake.petsc import PETSc
from mpi4py import MPI

from perphil.experiments.iterative_bench import (
    Approach,
    build_mesh,
    build_spaces,
    default_bcs,
    default_model_params,
    solve_on_mesh,
)
from perphil.utils.manufactured_solutions import exact_expressions

# Prefer explicit execution for timing stability
fd.parameters["pyop2_options"]["lazy_evaluation"] = False

# Start PETSc logging as early as possible
_PETSC_LOG_STARTED = False
try:
    PETSc.Log.begin()
    _PETSC_LOG_STARTED = True
except Exception:
    _PETSC_LOG_STARTED = False


def ensure_petsc_logging() -> None:
    global _PETSC_LOG_STARTED
    if not _PETSC_LOG_STARTED:
        PETSc.Log.begin()
        _PETSC_LOG_STARTED = True


# Logical -> possible PETSc names (varies by version)
EVENT_ALIASES: Dict[str, List[str]] = {
    "SNESSolve": ["SNESSolve", "SNES_Solve"],
    "SNESFunctionEval": ["SNESFunctionEval", "SNES_FunctionEval"],
    "SNESJacobianEval": ["SNESJacobianEval", "SNES_JacobianEval"],
    "KSPSolve": ["KSPSolve", "KSP_Solve"],
    "PCSetUp": ["PCSetUp", "PC_SetUp"],
    "PCApply": ["PCApply", "PC_Apply"],
    "MatMult": ["MatMult", "Mat_Mult"],
    "MatAssemblyBegin": ["MatAssemblyBegin", "Mat_AssemblyBegin"],
    "MatAssemblyEnd": ["MatAssemblyEnd", "Mat_AssemblyEnd"],
    "KSPGMRESOrthogonalization": ["KSPGMRESOrthogonalization", "KSP_GMRESOrthogonalization"],
    "KSPGMRESBuildBasis": ["KSPGMRESBuildBasis", "KSP_GMRESBuildBasis"],
}
DEFAULT_LOGICAL_EVENTS = [
    "KSPSolve",
    "PCApply",
    "PCSetUp",
    "MatMult",
    "MatAssemblyBegin",
    "MatAssemblyEnd",
    "SNESSolve",
    "SNESFunctionEval",
    "SNESJacobianEval",
]


def _match_event(name: str, logical_events: List[str]) -> Optional[str]:
    for logical in logical_events:
        for alias in EVENT_ALIASES.get(logical, [logical]):
            if name == alias or name.replace(" ", "") == alias.replace(" ", ""):
                return logical
    return None


def _reduce_avg(comm: MPI.Comm, data: Dict[str, float]) -> Dict[str, float]:
    return {k: comm.allreduce(v, op=MPI.SUM) / comm.size for k, v in data.items()}


def _parse_petsc_json(
    path: str, logical_events: List[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Parse PETSc -log_view JSON. Return (times_s, flops) per logical event.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            raise ValueError("Empty PETSc JSON log")
        data = json.loads(content)

    times: Dict[str, float] = {e: 0.0 for e in logical_events}
    flops: Dict[str, float] = {e: 0.0 for e in logical_events}

    def add_event(ev: Dict[str, Any]) -> None:
        name = ev.get("name", "")
        t = float(ev.get("time", 0.0))
        f = float(ev.get("flops", ev.get("flop", 0.0)))
        logical = _match_event(name, logical_events)
        if logical is not None:
            times[logical] = times.get(logical, 0.0) + t
            flops[logical] = flops.get(logical, 0.0) + f

    stages = data.get("stages") or []
    if stages:
        for st in stages:
            for ev in st.get("events", []):
                add_event(ev)
    elif "events" in data:
        for ev in data.get("events", []):
            add_event(ev)
    else:
        raise ValueError("Unrecognized PETSc JSON schema")
    return times, flops


# Try to capture Time and optional Flop column (ASCII -log_view)
ASCII_EVENT_LINE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_ /-]+?)\s{2,}(\d+)\s+([0-9.+\-eE]+)(?:\s+([0-9.+\-eE]+))?"
)


def _parse_petsc_ascii_file(
    path: str, logical_events: List[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Parse ASCII -log_view output (one file, one rank). Return (times_s, flops).
    """
    times: Dict[str, float] = {e: 0.0 for e in logical_events}
    flops: Dict[str, float] = {e: 0.0 for e in logical_events}

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return times, flops

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    in_table = False
    for line in lines:
        if "Event" in line and "Time (sec)" in line and "Count" in line:
            in_table = True
            continue
        if in_table:
            if not line.strip() or set(line.strip()) <= {"-", "="}:
                in_table = False
                continue
            m = ASCII_EVENT_LINE.match(line)
            if m:
                raw_name = m.group(1).strip()
                t = float(m.group(3))
                f = float(m.group(4)) if m.group(4) is not None else 0.0
                logical = _match_event(raw_name, logical_events)
                if logical is not None:
                    times[logical] = times.get(logical, 0.0) + t
                    flops[logical] = flops.get(logical, 0.0) + f
    return times, flops


class _StageCtx:
    """Fallback stage API reader."""

    def __init__(self, name: str):
        self.stage = PETSc.Log.Stage(name)

    def __enter__(self):
        self.stage.push()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stage.pop()

    def times_and_flops(
        self, logical_events: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        t_out: Dict[str, float] = {}
        f_out: Dict[str, float] = {}
        for logical in logical_events:
            t_sum = 0.0
            f_sum = 0.0
            for alias in EVENT_ALIASES.get(logical, [logical]):
                try:
                    ev = PETSc.Log.Event(alias)
                    info = self.stage.getEventPerfInfo(ev)
                    t_sum += float(info.get("time", 0.0))
                    f_sum += float(info.get("flops", info.get("flop", 0.0)))
                except Exception:
                    pass
            t_out[logical] = t_sum
            f_out[logical] = f_sum
        return t_out, f_out


def _snapshot_events(logical_events: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Snapshot current PETSc per-event totals (time, flops) aggregated across stages.
    """
    times: Dict[str, float] = {e: 0.0 for e in logical_events}
    flops: Dict[str, float] = {e: 0.0 for e in logical_events}
    for logical in logical_events:
        t_sum = 0.0
        f_sum = 0.0
        for alias in EVENT_ALIASES.get(logical, [logical]):
            try:
                ev = PETSc.Log.Event(alias)
                info = ev.getPerfInfo()
                t_sum += float(info.get("time", 0.0))
                f_sum += float(info.get("flops", info.get("flop", 0.0)))
            except Exception:
                pass
        times[logical] = t_sum
        flops[logical] = f_sum
    return times, flops


def _profile_with_events_api(
    comm: MPI.Comm,
    run_fn: Callable[[], Any],
    logical_events: List[str],
    repeats: int,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Use PETSc.Log.Event.getPerfInfo() before/after the run and take differences.
    """
    ensure_petsc_logging()
    comm.barrier()
    before_t, before_f = _snapshot_events(logical_events)
    t0 = time.perf_counter()
    for _ in range(max(1, repeats)):
        run_fn()
    wall = time.perf_counter() - t0
    comm.barrier()
    after_t, after_f = _snapshot_events(logical_events)

    local_t = {
        k: max(0.0, after_t.get(k, 0.0) - before_t.get(k, 0.0))
        for k in set(before_t) | set(after_t)
    }
    local_f = {
        k: max(0.0, after_f.get(k, 0.0) - before_f.get(k, 0.0))
        for k in set(before_f) | set(after_f)
    }

    times = _reduce_avg(comm, local_t)
    flops = _reduce_avg(comm, local_f)
    return times, flops, wall


def _profile_with_log_view_json(
    comm: MPI.Comm,
    run_fn: Callable[[], Any],
    logical_events: List[str],
    repeats: int,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    handle, path = tempfile.mkstemp(suffix=".json", prefix="petsc_log_")
    os.close(handle)
    try:
        opts = PETSc.Options()
        opts["log_view"] = f":{path}:json"
        t0 = time.perf_counter()
        for _ in range(max(1, repeats)):
            run_fn()
        wall = time.perf_counter() - t0
        PETSc.Log.view()
        local_times, local_flops = _parse_petsc_json(path, logical_events)
        times = _reduce_avg(comm, local_times)
        flops = _reduce_avg(comm, local_flops)
        return times, flops, wall
    finally:
        try:
            PETSc.Options().delValue("log_view")
        except Exception:
            pass
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def _profile_with_log_view_ascii(
    comm: MPI.Comm,
    run_fn: Callable[[], Any],
    logical_events: List[str],
    repeats: int,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Use ASCII viewer; each rank writes its own file to avoid interleaving.
    """
    handle, base_path = tempfile.mkstemp(suffix=".log", prefix="petsc_log_")
    os.close(handle)
    os.remove(base_path)  # we will create rank-specific files

    rank = comm.rank
    path = f"{base_path}.rank{rank}"
    try:
        t0 = time.perf_counter()
        for _ in range(max(1, repeats)):
            run_fn()
        wall = time.perf_counter() - t0

        viewer = PETSc.Viewer().createASCII(path, comm=PETSc.COMM_SELF)
        PETSc.Log.view(viewer)
        viewer.destroy()

        local_times, local_flops = _parse_petsc_ascii_file(path, logical_events)
        times = _reduce_avg(comm, local_times)
        flops = _reduce_avg(comm, local_flops)
        return times, flops, wall
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def _profile_with_stage_api(
    comm: MPI.Comm,
    run_fn: Callable[[], Any],
    logical_events: List[str],
    repeats: int,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    st = _StageCtx("perphil_solve")
    with st:
        before_t, before_f = st.times_and_flops(logical_events)
        t0 = time.perf_counter()
        for _ in range(max(1, repeats)):
            run_fn()
        wall = time.perf_counter() - t0
        after_t, after_f = st.times_and_flops(logical_events)
    local_t = {
        k: max(0.0, after_t.get(k, 0.0) - before_t.get(k, 0.0))
        for k in set(before_t) | set(after_t)
    }
    local_f = {
        k: max(0.0, after_f.get(k, 0.0) - before_f.get(k, 0.0))
        for k in set(before_f) | set(after_f)
    }
    times = _reduce_avg(comm, local_t)
    flops = _reduce_avg(comm, local_f)
    return times, flops, wall


def _get_rss_kb() -> int:
    """
    Peak resident set size in KB (normalized). Linux ru_maxrss is KB; macOS is bytes.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Heuristic: normalize to KB if value looks like bytes (macOS)
    return int(rss // 1024) if rss > 10_000_000 else int(rss)


def _collect_matrix_memory(solver_obj: Any) -> Dict[str, float]:
    """
    Collect matrix memory stats for A/P and factor (if available).
    Returns keys: mat_nz_used, mat_nz_allocated, mat_memory_mb, pmat_*, factor_*.
    """
    out: Dict[str, float] = {}

    def add_mat(prefix: str, M: Optional[PETSc.Mat]) -> None:
        if M is None:
            return
        try:
            info = M.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)
            nz_used = float(info.get("nz_used", 0.0))
            nz_alloc = float(info.get("nz_allocated", 0.0))
            mem_b = float(info.get("memory", 0.0))
            out[f"{prefix}_nz_used"] = nz_used
            out[f"{prefix}_nz_allocated"] = nz_alloc
            out[f"{prefix}_memory_mb"] = mem_b / (1024.0 * 1024.0)
        except Exception:
            pass

    try:
        # Get KSP handle
        ksp = None
        snes = getattr(solver_obj, "snes", None)
        if snes is not None and hasattr(snes, "ksp"):
            ksp = snes.ksp
        elif hasattr(solver_obj, "ksp"):
            ksp = solver_obj.ksp
        if ksp is None:
            return out

        # Operators A and P
        try:
            A, P = ksp.getOperators()
        except Exception:
            A = P = None
        add_mat("mat", A)
        add_mat("pmat", P)

        # Factor matrix for LU/ILU
        try:
            pc = ksp.getPC()
            pct = (pc.getType() or "").lower()
            if "lu" in pct or "ilu" in pct:
                try:
                    F = pc.getFactorMatrix()
                    add_mat("factor", F)
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass
    return out


@dataclass
class PerfResult:
    approach: str
    nx: int
    ny: int
    dofs: int
    num_cells: int
    iterations: Optional[int]
    residual: float
    times: Dict[str, float]
    flops: Dict[str, float]
    metadata: Dict[str, Any]
    memory: Optional[Dict[str, float]] = None
    # Universal wall-clock metrics
    time_total: float = 0.0  # average wall time per run (seconds)
    time_total_repeats: float = 0.0  # total wall time across repeats (seconds)

    def to_dict(self) -> Dict[str, Any]:
        base = asdict(self)
        # Flatten times
        for k, v in self.times.items():
            base[f"time_{k}"] = v
        # Flatten flops and mflops/s
        for k, v in self.flops.items():
            base[f"flops_{k}"] = v
            t = self.times.get(k, 0.0)
            base[f"mflops_{k}"] = (v / t / 1e6) if t > 0.0 else 0.0
        base["flops_total"] = float(sum(self.flops.values()))
        # Flatten memory
        if self.memory:
            for k, v in self.memory.items():
                base[f"mem_{k}"] = v
        # Universal time
        base["time_total"] = float(self.time_total)
        base["time_total_repeats"] = float(self.time_total_repeats)
        # Drop nested dicts
        base.pop("times", None)
        base.pop("flops", None)
        base.pop("memory", None)
        return base


def _enable_convergence_history_if_possible(solver_obj: Any) -> None:
    try:
        snes = getattr(solver_obj, "snes", None)
        if snes is not None and hasattr(snes, "setConvergenceHistory"):
            snes.setConvergenceHistory()
            try:
                snes.ksp.setConvergenceHistory()
            except Exception:
                pass
            return
        ksp = getattr(solver_obj, "ksp", None)
        if ksp is not None and hasattr(ksp, "setConvergenceHistory"):
            ksp.setConvergenceHistory()
    except Exception:
        pass


def _extract_ksp_iters_if_possible(solver_obj: Any) -> Optional[int]:
    try:
        snes = getattr(solver_obj, "snes", None)
        if snes is not None and hasattr(snes, "ksp"):
            return int(snes.ksp.getIterationNumber())
        ksp = getattr(solver_obj, "ksp", None)
        if ksp is not None and hasattr(ksp, "getIterationNumber"):
            return int(ksp.getIterationNumber())
    except Exception:
        return None
    return None


def _extract_solution_handle(sol: Any) -> Any:
    for key in ("petsc_solver", "solver", "petsc_snes", "petsc_ksp"):
        if hasattr(sol, key):
            return getattr(sol, key)
    return sol


def run_perf_once(
    nx: int,
    ny: int,
    approach: Approach,
    eager: bool = True,
    logical_events: Optional[List[str]] = None,
    force_nonzero_rhs: bool = False,
    bc_values: Optional[List[float]] = None,
    repeats: int = 5,
    backend: str = "auto",  # "auto" | "json" | "ascii" | "events" | "stage" | "wall"
    use_manufactured: bool = True,  # apply manufactured pressures as Dirichlet BCs
) -> PerfResult:
    """
    Run one profiled solve (optionally repeated) and return PETSc event times and flops.
    """
    ensure_petsc_logging()

    mesh = build_mesh(nx, ny, quadrilateral=True)
    _, _, W = build_spaces(mesh)
    comm = mesh.comm

    # Model parameters
    params = default_model_params()

    # BCs
    if use_manufactured:
        # Manufactured pressures as boundary conditions
        _u1e, p1e, _u2e, p2e = exact_expressions(mesh, params)
        bcs = [
            fd.DirichletBC(W.sub(0), p1e, "on_boundary"),
            fd.DirichletBC(W.sub(1), p2e, "on_boundary"),
        ]
    elif force_nonzero_rhs:
        v = bc_values or [1.0, 0.0]
        bcs = [
            fd.DirichletBC(W.sub(0), fd.Constant(v[0]), "on_boundary"),
            fd.DirichletBC(W.sub(1), fd.Constant(v[1]), "on_boundary"),
        ]
    else:
        bcs = default_bcs(W)

    logical_events = list(dict.fromkeys((logical_events or []) + DEFAULT_LOGICAL_EVENTS))

    # Warmup
    if eager:
        sol_warm = solve_on_mesh(W, approach, params=params, bcs=bcs)
        _enable_convergence_history_if_possible(_extract_solution_handle(sol_warm))

    # Runner closure
    def run_once():
        solve_on_mesh(W, approach, params=params, bcs=bcs)

    # Record RSS before profiling this case (per-rank), to compute delta
    rss_before_kb = _get_rss_kb()

    # Select backend(s)
    backends = [backend] if backend != "auto" else ["json", "ascii", "events", "stage", "wall"]

    wall_total = 0.0
    for be in backends:
        try:
            if be == "json":
                times, flops, wall = _profile_with_log_view_json(
                    comm, run_once, logical_events, repeats
                )
            elif be == "ascii":
                times, flops, wall = _profile_with_log_view_ascii(
                    comm, run_once, logical_events, repeats
                )
            elif be == "events":
                times, flops, wall = _profile_with_events_api(
                    comm, run_once, logical_events, repeats
                )
            elif be == "stage":
                times, flops, wall = _profile_with_stage_api(
                    comm, run_once, logical_events, repeats
                )
            else:
                t0 = time.perf_counter()
                for _ in range(max(1, repeats)):
                    run_once()
                wall = time.perf_counter() - t0
                times = {e: 0.0 for e in logical_events}
                times["KSPSolve"] = wall
                flops = {e: 0.0 for e in logical_events}
            # Accept if any event is nonzero or backend is "wall"/"events"
            if (sum(times.values()) > 0.0) or be in ("wall", "events"):
                backend_used = be
                wall_total = wall
                break
        except Exception:
            continue
    else:
        # If all failed, use wallclock only
        t0 = time.perf_counter()
        for _ in range(max(1, repeats)):
            run_once()
        wall = time.perf_counter() - t0
        times = {e: 0.0 for e in logical_events}
        times["KSPSolve"] = wall
        flops = {e: 0.0 for e in logical_events}
        backend_used = "wall"
        wall_total = wall

    # Solve once to collect iteration/residual and build operators for memory stats
    sol = solve_on_mesh(W, approach, params=params, bcs=bcs)

    # Metadata and counters
    num_cells = comm.allreduce(mesh.cell_set.size, op=MPI.SUM)
    dofs = W.dim()
    solver_handle = _extract_solution_handle(sol)
    ksp_iters = _extract_ksp_iters_if_possible(solver_handle)
    iterations = ksp_iters if ksp_iters is not None else getattr(sol, "iteration_number", None)
    residual = float(getattr(sol, "residual_error", np.nan))

    # Memory metrics
    rss_after_kb = _get_rss_kb()
    rss_peak_kb = float(comm.allreduce(rss_after_kb, op=MPI.MAX))
    rss_delta_kb = float(comm.allreduce(max(0, rss_after_kb - rss_before_kb), op=MPI.MAX))
    mat_mem = _collect_matrix_memory(solver_handle)
    memory: Dict[str, float] = {
        "rss_peak_kb": rss_peak_kb,
        "rss_delta_kb": rss_delta_kb,
        **mat_mem,
    }

    meta: Dict[str, Any] = {
        "petsc_version": PETSc.Sys.getVersion(),
        "firedrake_version": getattr(fd, "__version__", None),
        "backend": backend_used,
        "repeats": repeats,
    }

    return PerfResult(
        approach=approach.value,
        nx=nx,
        ny=ny,
        dofs=int(dofs),
        num_cells=int(num_cells),
        iterations=iterations,
        residual=residual,
        times=times,
        flops=flops,
        metadata=meta,
        memory=memory,
        time_total=(wall_total / max(1, repeats)),
        time_total_repeats=wall_total,
    )


def run_perf_sweep(
    mesh_sizes: List[int],
    approaches: List[Approach],
    logical_events: Optional[List[str]] = None,
    eager: bool = True,
    force_nonzero_rhs: bool = True,
    bc_values: Optional[List[float]] = None,
    repeats: int = 5,
    backend: str = "auto",
    use_manufactured: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for nx in mesh_sizes:
        ny = nx
        for ap in approaches:
            res = run_perf_once(
                nx,
                ny,
                ap,
                eager=eager,
                logical_events=logical_events,
                force_nonzero_rhs=force_nonzero_rhs,
                bc_values=bc_values,
                repeats=repeats,
                backend=backend,
                use_manufactured=use_manufactured,
            )
            rows.append(res.to_dict())
            PETSc.Sys.Print(
                f"[perf] nx={nx} {ap.value}: iters={res.iterations}, "
                f"time_total={res.time_total:.3e}s, "
                f"KSPSolve={res.times.get('KSPSolve', 0):.3e}s, "
                f"PCApply={res.times.get('PCApply', 0):.3e}s, "
                f"flops_total={sum(res.flops.values()):.3e} "
                f"(backend={res.metadata.get('backend')}, repeats={repeats})"
            )
    return pd.DataFrame(rows)


def save_perf_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def save_perf_json(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
