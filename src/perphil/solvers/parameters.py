_MAX_ITERATION_NUMBER = 50000

# Linear monolithic solver parameters (direct solver via MUMPS)
LINEAR_SOLVER_PARAMS: dict = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# Pure GMRES parameters
GMRES_PARAMS: dict = {
    "mat_type": "aij",
    "ksp_type": "gmres",
    "ksp_rtol": 1.0e-8,
    "ksp_atol": 1.0e-12,
    "ksp_max_it": _MAX_ITERATION_NUMBER,
}

# Plain GMRES without preconditioners
PLAIN_GMRES_PARAMS: dict = {"pc_type": "none", **GMRES_PARAMS}

# GMRES + Jacobi parameters for scale-splitting comparison
GMRES_JACOBI_PARAMS: dict = {"pc_type": "jacobi", **GMRES_PARAMS}

# GMRES + ILU additional parameters for scale-splitting comparison
GMRES_ILU_PARAMS: dict = {"pc_type": "ilu", "pc_factor_levels": 0, **GMRES_PARAMS}

# Field-split preconditioner (multiplicative) with LU in each block
FIELDSPLIT_LU_PARAMS: dict = {
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "pc_fieldsplit_0_fields": "0",
    "pc_fieldsplit_1_fields": "1",
    "fieldsplit_0": LINEAR_SOLVER_PARAMS,
    "fieldsplit_1": LINEAR_SOLVER_PARAMS,
}

# Field-split preconditioner (multiplicative) with GMRES in each block
FIELDSPLIT_GMRES_PARAMS: dict = {
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "pc_fieldsplit_0_fields": "0",
    "pc_fieldsplit_1_fields": "1",
    "fieldsplit_0": PLAIN_GMRES_PARAMS,
    "fieldsplit_1": PLAIN_GMRES_PARAMS,
}

# Field-split preconditioner (multiplicative) with GMRES + ILU in each block
FIELDSPLIT_GMRES_ILU_PARAMS: dict = {
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "pc_fieldsplit_0_fields": "0",
    "pc_fieldsplit_1_fields": "1",
    "fieldsplit_0": GMRES_ILU_PARAMS,
    "fieldsplit_1": GMRES_ILU_PARAMS,
}

# Picard (nonlinear Richardson) solver parameters with field-split
RICHARDSON_SOLVER_PARAMS: dict = {
    "snes_type": "nrichardson",
    "snes_max_it": _MAX_ITERATION_NUMBER,
    "snes_linesearch_type": "basic",
    "snes_linesearch_damping": 0.5,
    "snes_rtol": 1e-5,
    "snes_atol": 1e-12,
    **FIELDSPLIT_LU_PARAMS,
}

# Picard (with nonlinear Gauss-Siedel and LU) solver parameters with field-split
PICARD_LU_SOLVER_PARAMS = {
    "snes_type": "ngs",
    "snes_max_it": _MAX_ITERATION_NUMBER,
    "snes_rtol": 1e-8,
    "snes_atol": 1e-12,
    **FIELDSPLIT_LU_PARAMS,
}

# Picard (with GMRES) solver parameters with field-split
PICARD_GMRES_SOLVER_PARAMS = {
    "snes_type": "ngs",
    "snes_max_it": _MAX_ITERATION_NUMBER,
    "snes_rtol": 1e-8,
    "snes_atol": 1e-12,
    **FIELDSPLIT_GMRES_PARAMS,
}

# Picard (with GMRES + ILU) solver parameters with field-split
PICARD_GMRES_ILU_SOLVER_PARAMS = {
    "snes_type": "ngs",
    "snes_max_it": _MAX_ITERATION_NUMBER,
    "snes_rtol": 1e-8,
    "snes_atol": 1e-12,
    **FIELDSPLIT_GMRES_ILU_PARAMS,
}

# SNES with KSP-only (for preconditioner analysis)
KSP_PREONLY_PARAMS: dict = {
    "snes_type": "ksponly",
    "ksp_monitor": None,
    **FIELDSPLIT_LU_PARAMS,
}
