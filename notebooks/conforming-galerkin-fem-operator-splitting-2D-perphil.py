# %% [markdown]
# # 2D DPP conforming Galerkin FEM
#
# Exploration notebook with all possible approaches, checking individually if the results are close to the exact solution.

# %%
import os
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"

import firedrake as fd

from perphil.forms.spaces import create_function_spaces
from perphil.forms.dpp import dpp_form, dpp_delayed_form
from perphil.mesh.builtin import create_mesh
from perphil.models.dpp.parameters import DPPParameters
from perphil.solvers.conditioning import (
    get_matrix_data_from_form,
    calculate_condition_number,
)
from perphil.solvers.solver import (
    solve_dpp,
    solve_dpp_nonlinear,
)
from perphil.solvers.parameters import (
    LINEAR_SOLVER_PARAMS,
    GMRES_PARAMS,
    FIELDSPLIT_LU_PARAMS,
    GMRES_ILU_PARAMS,
    PICARD_LU_SOLVER_PARAMS,
    PICARD_GMRES_SOLVER_PARAMS,
    PICARD_GMRES_ILU_SOLVER_PARAMS,
    PLAIN_GMRES_PARAMS,
)
from perphil.utils.plotting import plot_2d_mesh, plot_scalar_field, plot_vector_field
from perphil.utils.manufactured_solutions import interpolate_exact
from perphil.utils.postprocessing import (
    split_dpp_solution,
    calculate_darcy_velocity_from_pressure,
    slice_along_x,
)

# %% [markdown]
# For convenience, we define the operators from Firedrake:

# %%
grad = fd.grad
div = fd.div
dx = fd.dx
inner = fd.inner
pi = fd.pi
sin = fd.sin
exp = fd.exp
cos = fd.cos

# %% [markdown]
# ## Case 1

# %% [markdown]
# ### Mesh

# %%
mesh = create_mesh(10, 10, quadrilateral=True)

# %%
plot_2d_mesh(mesh)

# %% [markdown]
# ### Exact solutions

# %%
U, V = create_function_spaces(
    mesh,
    velocity_deg=1,
    pressure_deg=1,
    velocity_family="CG",
    pressure_family="CG",
)

dpp_params = DPPParameters(k1=1.0, k2=1 / 1e2, beta=1.0, mu=1)
u1_exact, p1_exact, u2_exact, p2_exact = interpolate_exact(mesh, U, V, dpp_params)

# %%
plot_scalar_field(p1_exact)
plot_scalar_field(p2_exact)
plot_vector_field(u1_exact)
plot_vector_field(u2_exact)

# %% [markdown]
# ### Conforming Galerkin FEM approximations

# %% [markdown]
# #### Monolithic (fully coupled) approximation

# %%
W = V * V  # Mixed function space with both scales

# Dirichlet BCs
bc_macro = fd.DirichletBC(W.sub(0), p1_exact, "on_boundary")
bc_micro = fd.DirichletBC(W.sub(1), p2_exact, "on_boundary")
bcs = [bc_macro, bc_micro]

solver_parameters = LINEAR_SOLVER_PARAMS
solution_data_monolithic = solve_dpp(
    W, dpp_params, bcs, solver_parameters=solver_parameters
)
solution_monolithic = solution_data_monolithic.solution
p1_monolithic, p2_monolithic = split_dpp_solution(solution_monolithic)

u1_monolithic = calculate_darcy_velocity_from_pressure(p1_monolithic, dpp_params.k1)

u2_monolithic = calculate_darcy_velocity_from_pressure(p2_monolithic, dpp_params.k2)

# %%
plot_scalar_field(p1_monolithic, title=r"$p_1$ scalar field")
plot_scalar_field(p2_monolithic, title=r"$p_2$ scalar field")
plot_vector_field(u1_monolithic, title=r"$u_1$ vector field")
plot_vector_field(u2_monolithic, title=r"$u_2$ vector field")

# %%
x_mid_point = 0.5
y_points, p1_mono_at_x_mid_point = slice_along_x(p1_monolithic, x_value=x_mid_point)
_, p1_exact_at_x_mid_point = slice_along_x(p1_exact, x_value=x_mid_point)
_, p2_mono_at_x_mid_point = slice_along_x(p2_monolithic, x_value=x_mid_point)
_, p2_exact_at_x_mid_point = slice_along_x(p2_exact, x_value=x_mid_point)

y_points, p1_mono_at_x_mid_point, p2_mono_at_x_mid_point

# %%
figsize = (7, 7)
plt.figure(figsize=figsize)
plt.plot(
    y_points, p1_mono_at_x_mid_point, "x", ms=10, lw=4, c="k", label="Monolithic LU"
)
plt.plot(y_points, p1_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Macro Pressure $(p_{1,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

plt.figure(figsize=figsize)
plt.plot(
    y_points, p2_mono_at_x_mid_point, "x", ms=10, lw=4, c="k", label="Monolithic LU"
)
plt.plot(y_points, p2_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Micro Pressure $(p_{2,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

# %% [markdown]
# #### Monolithic (fully coupled) approximation with plain GMRES

# %%
solver_additional_param = {
    "ksp_monitor": None,
    "snes_monitor": None,
    "snes_rtol": 1e-8,
    "snes_atol": 1e-12,
}
solver_parameters = {**PLAIN_GMRES_PARAMS, **solver_additional_param}
solution_data_monolithic_gmres = solve_dpp(
    W, dpp_params, bcs, solver_parameters=solver_parameters
)
solution_monolithic_gmres = solution_data_monolithic_gmres.solution
p1_gmres, p2_gmres = split_dpp_solution(solution_monolithic_gmres)

u1_gmres = calculate_darcy_velocity_from_pressure(p1_gmres, dpp_params.k1)

u2_gmres = calculate_darcy_velocity_from_pressure(p2_gmres, dpp_params.k2)

# %%
y_points, p1_gmres_at_x_mid_point = slice_along_x(p1_gmres, x_value=x_mid_point)
_, p2_gmres_at_x_mid_point = slice_along_x(p2_gmres, x_value=x_mid_point)

y_points, p1_gmres_at_x_mid_point, p2_gmres_at_x_mid_point

# %%
figsize = (7, 7)
plt.figure(figsize=figsize)
plt.plot(y_points, p1_gmres_at_x_mid_point, "x", ms=10, lw=4, c="k", label="GMRES")
plt.plot(y_points, p1_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Macro Pressure $(p_{1,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

plt.figure(figsize=figsize)
plt.plot(y_points, p2_gmres_at_x_mid_point, "x", ms=10, lw=4, c="k", label="GMRES")
plt.plot(y_points, p2_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Micro Pressure $(p_{2,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

# %% [markdown]
# #### Monolithic approximation with GMRES + ILU

# %%
solver_parameters = {**GMRES_ILU_PARAMS, **solver_additional_param}
solution_data_monolithic_gmres_ilu = solve_dpp(
    W, dpp_params, bcs, solver_parameters=solver_parameters
)
solution_monolithic_gmres_ilu = solution_data_monolithic_gmres_ilu.solution
p1_gmres_ilu, p2_gmres_ilu = split_dpp_solution(solution_monolithic_gmres_ilu)

u1_gmres_ilu = calculate_darcy_velocity_from_pressure(p1_gmres_ilu, dpp_params.k1)

u2_gmres_ilu = calculate_darcy_velocity_from_pressure(p2_gmres_ilu, dpp_params.k2)

# %%
y_points, p1_gmres_ilu_at_x_mid_point = slice_along_x(p1_gmres_ilu, x_value=x_mid_point)
_, p2_gmres_ilu_at_x_mid_point = slice_along_x(p2_gmres_ilu, x_value=x_mid_point)

y_points, p1_gmres_ilu_at_x_mid_point, p2_gmres_ilu_at_x_mid_point

# %%
figsize = (7, 7)
plt.figure(figsize=figsize)
plt.plot(
    y_points, p1_gmres_ilu_at_x_mid_point, "x", ms=10, lw=4, c="k", label="GMRES + ILU"
)
plt.plot(y_points, p1_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Macro Pressure $(p_{1,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

plt.figure(figsize=figsize)
plt.plot(
    y_points, p2_gmres_ilu_at_x_mid_point, "x", ms=10, lw=4, c="k", label="GMRES + ILU"
)
plt.plot(y_points, p2_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Micro Pressure $(p_{2,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

# %% [markdown]
# #### Scale-splitting

# %% [markdown]
# Pre-conditioner by scale:

# %%
solver_parameters = {**GMRES_PARAMS, **FIELDSPLIT_LU_PARAMS, **solver_additional_param}
solution_data_monolithic_gmres = solve_dpp(
    W, dpp_params, bcs, solver_parameters=solver_parameters
)
solution_preconditioned = solution_data_monolithic_gmres.solution
p1_gmres, p2_preconditioned = split_dpp_solution(solution_preconditioned)

u1_preconditioned = calculate_darcy_velocity_from_pressure(p1_gmres, dpp_params.k1)

u2_preconditioned = calculate_darcy_velocity_from_pressure(
    p2_preconditioned, dpp_params.k2
)

# %%
solver_parameters = {
    **GMRES_ILU_PARAMS,
    **FIELDSPLIT_LU_PARAMS,
    **solver_additional_param,
}
solution_data_gmres_ilu = solve_dpp(
    W, dpp_params, bcs, solver_parameters=solver_parameters
)
solution_gmres_ilu = solution_data_gmres_ilu.solution
p1_gmres_ilu, p2_gmres_ilu = split_dpp_solution(solution_gmres_ilu)

u1_gmres_ilu = calculate_darcy_velocity_from_pressure(p1_gmres_ilu, dpp_params.k1)

u2_gmres_ilu = calculate_darcy_velocity_from_pressure(p2_gmres_ilu, dpp_params.k2)

# %%
y_points, p1_pc_at_x_mid_point = slice_along_x(p1_gmres, x_value=x_mid_point)
_, p2_pc_at_x_mid_point = slice_along_x(p2_preconditioned, x_value=x_mid_point)

y_points, p1_pc_at_x_mid_point, p2_pc_at_x_mid_point

# %%
figsize = (7, 7)
plt.figure(figsize=figsize)
plt.plot(y_points, p1_pc_at_x_mid_point, "x", ms=10, lw=4, c="k", label="Fieldsplit PC")
plt.plot(y_points, p1_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Macro Pressure $(p_{1,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

plt.figure(figsize=figsize)
plt.plot(y_points, p2_pc_at_x_mid_point, "x", ms=10, lw=4, c="k", label="Fieldsplit PC")
plt.plot(y_points, p2_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Micro Pressure $(p_{2,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

# %% [markdown]
# Loop-based Picard fixed-point iterations using inner LU solver:

# %%
solver_parameters = {**PICARD_LU_SOLVER_PARAMS, **solver_additional_param}
solution_data_ngs = solve_dpp_nonlinear(
    W, dpp_params, bcs, solver_parameters=solver_parameters
)
solution_ngs = solution_data_ngs.solution
p1_picard, p2_picard = split_dpp_solution(solution_ngs)

u1_picard = calculate_darcy_velocity_from_pressure(p1_picard, dpp_params.k1)

u2_picard = calculate_darcy_velocity_from_pressure(p2_picard, dpp_params.k2)

# %%
y_points, p1_picard_at_x_mid_point = slice_along_x(p1_picard, x_value=x_mid_point)
_, p2_picard_at_x_mid_point = slice_along_x(p2_picard, x_value=x_mid_point)

y_points, p1_picard_at_x_mid_point, p2_picard_at_x_mid_point

# %%
figsize = (7, 7)
plt.figure(figsize=figsize)
plt.plot(
    y_points,
    p1_picard_at_x_mid_point,
    "x",
    ms=10,
    lw=4,
    c="k",
    label="Loop-based Picard",
)
plt.plot(y_points, p1_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Macro Pressure $(p_{1,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

plt.figure(figsize=figsize)
plt.plot(
    y_points, p2_pc_at_x_mid_point, "x", ms=10, lw=4, c="k", label="Loop-based Picard"
)
plt.plot(y_points, p2_picard_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Micro Pressure $(p_{2,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

# %% [markdown]
# Loop-based Picard fixed-point iterations using inner GMRES solver:

# %%
solver_parameters = {**PICARD_GMRES_SOLVER_PARAMS, **solver_additional_param}
solution_data_picard_gmres = solve_dpp_nonlinear(
    W, dpp_params, bcs, solver_parameters=solver_parameters
)
solution_picard_gmres = solution_data_picard_gmres.solution
p1_picard_gmres, p2_picard_gmres = split_dpp_solution(solution_picard_gmres)

u1_picard_gmres = calculate_darcy_velocity_from_pressure(p1_picard_gmres, dpp_params.k1)

u2_picard_gmres = calculate_darcy_velocity_from_pressure(p2_picard_gmres, dpp_params.k2)

# %%
y_points, p1_picard_gmres_at_x_mid_point = slice_along_x(
    p1_picard_gmres, x_value=x_mid_point
)
_, p2_picard_gmres_at_x_mid_point = slice_along_x(p2_picard_gmres, x_value=x_mid_point)

y_points, p1_picard_gmres_at_x_mid_point, p2_picard_gmres_at_x_mid_point

# %%
figsize = (7, 7)
plt.figure(figsize=figsize)
plt.plot(
    y_points,
    p1_picard_gmres_at_x_mid_point,
    "x",
    ms=10,
    lw=4,
    c="k",
    label="Loop-based Picard (GMRES)",
)
plt.plot(y_points, p1_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Macro Pressure $(p_{1,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

plt.figure(figsize=figsize)
plt.plot(
    y_points,
    p2_picard_gmres_at_x_mid_point,
    "x",
    ms=10,
    lw=4,
    c="k",
    label="Loop-based Picard (GMRES)",
)
plt.plot(y_points, p2_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Micro Pressure $(p_{2,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

# %% [markdown]
# Loop-based Picard fixed-point iterations using inner GMRES + ILU solver:

# %%
solver_parameters = {**PICARD_GMRES_ILU_SOLVER_PARAMS, **solver_additional_param}
solution_data_picard_gmres_ilu = solve_dpp_nonlinear(
    W, dpp_params, bcs, solver_parameters=solver_parameters
)
solution_picard_gmres_ilu = solution_data_picard_gmres_ilu.solution
p1_picard_gmres_ilu, p2_picard_gmres_ilu = split_dpp_solution(solution_picard_gmres_ilu)

u1_picard_gmres_ilu = calculate_darcy_velocity_from_pressure(
    p1_picard_gmres_ilu, dpp_params.k1
)

u2_picard_gmres_ilu = calculate_darcy_velocity_from_pressure(
    p2_picard_gmres_ilu, dpp_params.k2
)

# %% [markdown]
# #### Conditioning Analysis

# %% [markdown]
# ##### Monolithic system

# %%
monolithic_lhs_form, _ = dpp_form(W=W, model_params=dpp_params)
matrix_data = get_matrix_data_from_form(monolithic_lhs_form, boundary_conditions=bcs)
monolithic_system_condition_number = calculate_condition_number(
    matrix_data.sparse_csr_data,
    num_of_factors=matrix_data.number_of_dofs - 1,
)
print(f"Monolithic system Condition Number: {monolithic_system_condition_number}")

# %% [markdown]
# ##### Scale-splitting

# %%
# Prepare initial zero estimates for pressures
p1_zero = fd.Function(V)
p1_zero.interpolate(fd.Constant(0.0))
p2_zero = fd.Function(V)
p2_zero.interpolate(fd.Constant(0.0))

# Build bilinear forms for scale-splitting
forms_macro, forms_micro = dpp_delayed_form(V, V, dpp_params, p1_zero, p2_zero)
a_macro_form, _ = forms_macro
a_micro_form, _ = forms_micro

# Create BCs for macro and micro in individual spaces
bc_macro_V = fd.DirichletBC(V, p1_exact, "on_boundary")
bc_micro_V = fd.DirichletBC(V, p2_exact, "on_boundary")

# Conditioning analysis for the macro system
matrix_data_macro = get_matrix_data_from_form(a_macro_form, [bc_macro_V])
macro_condition_number = calculate_condition_number(
    matrix_data_macro.sparse_csr_data,
    num_of_factors=matrix_data_macro.number_of_dofs - 1,
)
print(f"Macro system Condition Number: {macro_condition_number}")

# Conditioning analysis for the micro system
matrix_data_micro = get_matrix_data_from_form(a_micro_form, [bc_micro_V])
micro_condition_number = calculate_condition_number(
    matrix_data_micro.sparse_csr_data,
    num_of_factors=matrix_data_micro.number_of_dofs - 1,
)
print(f"Micro system Condition Number: {micro_condition_number}")
