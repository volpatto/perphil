"""
Experiments for benchmarking iterative and direct solvers for the Double Porosity/Permeability (DPP) model.

This module provides utilities to:
    - Build meshes and function spaces.
    - Configure solver parameters for different approaches.
    - Execute solvers and wrap results.
    - Assemble monolithic system matrices and estimate condition numbers.
    - Compute L2 errors against reference solutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.sparse as sp

import firedrake as fd

from perphil.mesh.builtin import create_mesh
from perphil.forms.spaces import create_function_spaces
from perphil.models.dpp.parameters import DPPParameters
from perphil.solvers.solver import solve_dpp, solve_dpp_nonlinear
from perphil.solvers import parameters as solver_params
from perphil.solvers import conditioning


class Approach(str, Enum):
    """
    Enumeration of solver approaches for the DPP model.

    - PLAIN_GMRES: Plain GMRES without specialized preconditioning.
    - GMRES_ILU: Plain GMRES with ILU preconditioning.
    - SS_GMRES: Scale-splitting GMRES with exact LU on block system.
    - SS_GMRES_ILU: Scale-splitting GMRES with ILU on block system.
    - PICARD_MUMPS: Nonlinear Picard iterations using MUMPS for linear subproblems.
    - MONOLITHIC_MUMPS: Direct monolithic solve using MUMPS.
    """

    PLAIN_GMRES = "GMRES"
    GMRES_ILU = "GMRES + ILU PC"
    SS_GMRES = "Scale-Splitting GMRES"
    SS_GMRES_ILU = "Scale-Splitting GMRES + ILU PC"
    PICARD_MUMPS = "Scaling-Splitting Picard with MUMPS"
    MONOLITHIC_MUMPS = "Monolithic LU with MUMPS"


@dataclass(frozen=True)
class SolveResult:
    """
    Data class to store results from solving the DPP problem.

    :param approach:
        The solver approach used.
    :param nx:
        Number of elements in the x direction (mesh resolution).
    :param ny:
        Number of elements in the y direction (mesh resolution).
    :param iteration_number:
        Number of iterations taken by the solver (if applicable).
    :param residual_error:
        Final residual error reported by the solver.
    :param fields:
        Optional tuple of Firedrake Functions (p1, p2) representing solution fields.
    """

    approach: Approach
    nx: int
    ny: int
    iteration_number: int
    residual_error: float
    # Optional: keep raw Firedrake Functions (mixed solution or split fields)
    fields: Optional[Tuple[fd.Function, fd.Function]] = None


def build_mesh(nx: int, ny: int, quadrilateral: bool = True) -> fd.Mesh:
    """
    Build a 2D unit-square mesh with specified element counts.

    :param nx:
        Number of elements in the x direction.
    :param ny:
        Number of elements in the y direction.
    :param quadrilateral:
        Whether to use quadrilateral elements (default True).
    :return:
        A Firedrake Mesh instance representing the unit square.
    """
    return create_mesh(nx, ny, quadrilateral=quadrilateral)


def build_spaces(mesh: fd.Mesh) -> Tuple[fd.FunctionSpace, fd.FunctionSpace, fd.MixedFunctionSpace]:
    """
    Build velocity, pressure, and mixed pressure function spaces.

    :param mesh:
        A Firedrake Mesh instance on which to build spaces.
    :return:
        A tuple (U, V, W) where U is velocity space, V is single pressure space,
        and W is mixed function space for (p1, p2).
    """
    _, V = create_function_spaces(mesh)
    W = fd.MixedFunctionSpace((V, V))
    return _, V, W


def default_bcs(W: fd.MixedFunctionSpace) -> List[fd.DirichletBC]:
    """
    Default homogeneous Dirichlet boundary conditions for both pressure fields.

    :param W:
        Mixed function space containing pressure subspaces.
    :return:
        A list of Firedrake DirichletBC objects for p1 and p2 on the boundary.
    """
    bc0 = fd.DirichletBC(W.sub(0), fd.Constant(0.0), "on_boundary")
    bc1 = fd.DirichletBC(W.sub(1), fd.Constant(0.0), "on_boundary")
    return [bc0, bc1]


def default_model_params() -> DPPParameters:
    """
    Create default DPP model parameters with unit values.

    :return:
        DPPParameters with k1=beta=mu=1.0, k2=k1/1e2.
    """
    return DPPParameters(k1=1.0, k2=1.0 / 1e2, beta=1.0, mu=1.0)


def make_fieldsplit_params_with(block_pc: str = "lu") -> Dict:
    """
    Build a field-split GMRES solver configuration with specified block preconditioners.

    :param block_pc:
        Preconditioner type for each block ('lu' or 'ilu').
    :return:
        A dictionary of PETSc solver parameters based on FIELDSPLIT_LU_PARAMS,
        configured for GMRES with block preconditioners.
    """
    base = dict(solver_params.FIELDSPLIT_LU_PARAMS)
    # Ensure outer solver is GMRES
    base["ksp_type"] = "gmres"
    # Switch sub-block preconditioners if requested
    if block_pc.lower() != "lu":
        base["fieldsplit_0_pc_type"] = block_pc
        base["fieldsplit_1_pc_type"] = block_pc
        # When using ILU, stick to preonly or small inner GMRES; preonly is fine here
        base["fieldsplit_0_ksp_type"] = base.get("fieldsplit_0_ksp_type", "preonly")
        base["fieldsplit_1_ksp_type"] = base.get("fieldsplit_1_ksp_type", "preonly")
    return base


def params_for(approach: Approach) -> Dict:
    """
    Return solver parameter dictionary for a given approach.

    :param approach:
        Approach enum specifying the solver strategy.
    :return:
        A dictionary of PETSc solver parameters appropriate for the approach.
    :raises ValueError:
        If the approach is unknown.
    """
    if approach == Approach.PLAIN_GMRES:
        # True plain GMRES baseline: explicitly disable preconditioning
        return solver_params.PLAIN_GMRES_PARAMS.copy()
    elif approach == Approach.GMRES_ILU:
        return solver_params.GMRES_ILU_PARAMS.copy()
    elif approach == Approach.SS_GMRES:
        PARAMS = {**solver_params.GMRES_PARAMS.copy(), **solver_params.FIELDSPLIT_LU_PARAMS.copy()}
        return PARAMS
    elif approach == Approach.SS_GMRES_ILU:
        PARAMS = {
            **solver_params.GMRES_PARAMS.copy(),
            **solver_params.FIELDSPLIT_GMRES_ILU_PARAMS.copy(),
        }
        return PARAMS
    elif approach == Approach.MONOLITHIC_MUMPS:
        return solver_params.LINEAR_SOLVER_PARAMS.copy()
    elif approach == Approach.PICARD_MUMPS:
        # Use the SNES-based NGS iterations for the Picard split form
        return solver_params.PICARD_LU_SOLVER_PARAMS.copy()
    else:
        raise ValueError(f"Unknown approach: {approach}")


def solve_on_mesh(
    W: fd.MixedFunctionSpace,
    approach: Approach,
    params: Optional[DPPParameters] = None,
    bcs: Optional[List[fd.DirichletBC]] = None,
) -> SolveResult:
    """
    Solve the DPP problem on a mixed function space using specified approach.

    :param W:
        MixedFunctionSpace for pressure fields.
    :param approach:
        Solver approach to use.
    :param params:
        Optional DPPParameters for the model. Defaults to unit parameters.
    :param bcs:
        Optional list of DirichletBC boundary conditions. Defaults to homogeneous.
    :return:
        SolveResult containing solver metadata and optional solution fields.
    """
    params = params or default_model_params()
    bcs = bcs or default_bcs(W)
    sp_dict = params_for(approach)

    if approach == Approach.PICARD_MUMPS:
        sol = solve_dpp_nonlinear(W, params, bcs=bcs, solver_parameters=sp_dict)
    else:
        sol = solve_dpp(W, params, bcs=bcs, solver_parameters=sp_dict)

    # Solution wrapper in repo exposes iteration_number and residual_error
    iters = getattr(sol, "iteration_number", -1)
    res = getattr(sol, "residual_error", np.nan)

    # Try to extract field Functions (p1, p2) for downstream error norms
    fields = None
    if hasattr(sol, "fields"):
        f = sol.fields
        # accept dict with keys or tuple/list
        if isinstance(f, dict):
            f1 = f.get("p1") or f.get(0)
            f2 = f.get("p2") or f.get(1)
            if f1 is not None and f2 is not None:
                fields = (f1, f2)
        elif isinstance(f, (tuple, list)) and len(f) == 2:
            fields = (f[0], f[1])
    if fields is None and hasattr(sol, "solution"):
        s = sol.solution
        # If solution is already a tuple/list of two fields, unpack directly
        if isinstance(s, (tuple, list)) and len(s) == 2:
            fields = (s[0], s[1])
        # Otherwise, if it’s a Firedrake Function, call split()
        elif hasattr(s, "split"):
            try:
                f1, f2 = s.split()
                fields = (f1, f2)
            except Exception:
                fields = None

    # nx,ny not known here; caller should fill these if needed
    return SolveResult(
        approach=approach, nx=-1, ny=-1, iteration_number=iters, residual_error=res, fields=fields
    )


def assemble_monolithic_matrix(
    W: fd.MixedFunctionSpace,
    params: Optional[DPPParameters] = None,
    bcs: Optional[List[fd.DirichletBC]] = None,
) -> Tuple[sp.csr_matrix, int, int]:
    """
    Assemble and extract the monolithic system matrix in CSR format.

    :param W:
        MixedFunctionSpace for the DPP problem.
    :param params:
        Optional DPPParameters for the model.
    :param bcs:
        Optional list of DirichletBC boundary conditions.
    :return:
        A tuple (csr, n0, n1) where csr is a scipy CSR matrix,
        and n0, n1 are the sizes of the macro and micro blocks.
    """
    params = params or default_model_params()
    bcs = bcs or default_bcs(W)

    # Build the bilinear form using repo's dpp_form
    # conditioning.get_matrix_data_from_form(form, bcs) → (mat, is_symmetric, csr, nnz, ndofs)
    from perphil.forms.dpp import dpp_form

    a, L = dpp_form(W, params)  # L is unused here; only matrix structure matters
    # Get matrix data and extract CSR from the MatrixData object
    md = conditioning.get_matrix_data_from_form(a, bcs)
    csr = md.sparse_csr_data

    n0 = W.sub(0).dim()
    n1 = W.sub(1).dim()
    return csr, n0, n1


def estimate_condition_numbers(
    W: fd.MixedFunctionSpace,
    params: Optional[DPPParameters] = None,
    bcs: Optional[List[fd.DirichletBC]] = None,
    num_of_factors: int = 50,
    use_sparse: bool = True,
) -> Dict[str, float]:
    """
    Compute condition numbers for the monolithic system and its macro/micro blocks.

    :param W:
        MixedFunctionSpace for the DPP problem.
    :param params:
        Optional DPPParameters for the model.
    :param bcs:
        Optional list of DirichletBC boundary conditions.
    :param num_of_factors:
        Number of Lanczos iterations to use for estimation.
    :param use_sparse:
        Whether to use sparse methods for condition number calculation.
    :return:
        A dict with keys 'monolithic', 'macro', and 'micro', mapping to their
        estimated condition numbers.
    """
    csr, n0, n1 = assemble_monolithic_matrix(W, params=params, bcs=bcs)

    # Monolithic cond(A)
    cond_full = conditioning.calculate_condition_number(
        csr, num_of_factors=num_of_factors, use_sparse=use_sparse
    )

    # Extract diagonal blocks
    A00 = csr[:n0, :n0]
    A11 = csr[n0 : n0 + n1, n0 : n0 + n1]

    # Ensure CSR
    A00 = A00.tocsr() if sp.issparse(A00) else sp.csr_matrix(A00)
    A11 = A11.tocsr() if sp.issparse(A11) else sp.csr_matrix(A11)

    cond_00 = conditioning.calculate_condition_number(
        A00, num_of_factors=num_of_factors, use_sparse=use_sparse
    )
    cond_11 = conditioning.calculate_condition_number(
        A11, num_of_factors=num_of_factors, use_sparse=use_sparse
    )

    return {"monolithic": cond_full, "macro": cond_00, "micro": cond_11}


def l2_errors_against_reference(
    W: fd.MixedFunctionSpace,
    fields: Tuple[fd.Function, fd.Function],
    ref_fields: Tuple[fd.Function, fd.Function],
) -> Tuple[float, float]:
    """
    Compute L2 error norms for each pressure field against a reference solution.

    :param W:
        MixedFunctionSpace for the DPP problem (used to access mesh).
    :param fields:
        Tuple (p1, p2) of computed pressure fields.
    :param ref_fields:
        Tuple (r1, r2) of reference pressure fields.
    :return:
        A tuple (e1, e2) of L2 error norms for fields p1 and p2.
    """
    p1, p2 = fields
    r1, r2 = ref_fields
    dx = fd.dx(domain=W.mesh())
    e1 = fd.sqrt(fd.assemble((p1 - r1) * (p1 - r1) * dx))
    e2 = fd.sqrt(fd.assemble((p2 - r2) * (p2 - r2) * dx))
    return float(e1), float(e2)
