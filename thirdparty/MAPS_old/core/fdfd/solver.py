# """
# Date: 2024-10-10 19:50:23
# LastEditors: Jiaqi Gu && jiaqigu@asu.edu
# LastEditTime: 2025-01-06 19:07:59
# FilePath: /MAPS/core/fdfd/solver.py
# """

# import cupy as cp
# import numpy as np
# import scipy.sparse.linalg as spl
# import torch
# from pyMKL import pardisoSolver
# from pyutils.general import logger
# from torch import Tensor

# from ...thirdparty.ceviche.constants import *
# from ...thirdparty.ceviche.utils import make_sparse
# from ...core.utils import print_stat
# import time
# try:
#     from pyMKL import pardisoSolver

#     HAS_MKL = True
#     # print('using MKL for direct solvers')
# except:
#     HAS_MKL = False


# def coo_torch2cupy(A):
#     A = A.data.coalesce()
#     Avals_cp = cp.asarray(A.values())
#     Aidx_cp = cp.asarray(A.indices())
#     # return cp.sparse.csr_matrix((Avals_cp, Aidx_cp), shape=A.shape)
#     return cp.sparse.coo_matrix((Avals_cp, Aidx_cp))


# # ---------------------- Sparse Solver Copied from Ceviche ----------------------
# DEFAULT_ITERATIVE_METHOD = "bicg"

# # dict of iterative methods supported (name: function)
# ITERATIVE_METHODS = {
#     "bicg": spl.bicg,
#     "bicgstab": spl.bicgstab,
#     "cg": spl.cg,
#     "cgs": spl.cgs,
#     "gmres": spl.gmres,
#     "lgmres": spl.lgmres,
#     "qmr": spl.qmr,
#     "gcrotmk": spl.gcrotmk,
# }

# # convergence tolerance for iterative solvers.
# ATOL = 1e-8

# """ ========================== SOLVER FUNCTIONS ========================== """


# def _solve_iterative_with_nn_init(
#     A, b, neural_solver, eps, iterative_method=DEFAULT_ITERATIVE_METHOD, **kwargs
# ):
#     """Iterative solver with neural network initial guess"""

#     # 1. Get initial guess from neural network
#     if neural_solver is not None:
#         # Convert b to a PyTorch tensor if it's a numpy array
#         if isinstance(b, np.ndarray):
#             b_torch = (
#                 torch.from_numpy(b).to(eps.device).to(torch.complex64)
#             )  # Ensure the tensor is on the same device as eps
#         else:
#             b_torch = b
#         x0 = neural_solver(
#             eps.unsqueeze(0),  # Add batch dimension
#             b_torch.reshape(eps.shape).unsqueeze(
#                 0
#             ),  # Reshape source term to match eps shape
#         )
#         if isinstance(x0, tuple):
#             x0 = x0[-1]  # Get final prediction if model has error correction
#         # Reshape NN output to match linear system size
#         x0 = x0.permute(0, 2, 3, 1).contiguous()
#         x0 = torch.view_as_complex(x0)
#         x0 = x0.squeeze(0).flatten()
#         x0 = x0.detach().cpu().numpy()
#         print("now we have the initial guess", flush=True)
#     else:
#         x0 = None  # Use default zero initial guess

#     # 2. Get the solver function
#     try:
#         solver_fn = ITERATIVE_METHODS[iterative_method]
#     except:
#         raise ValueError(
#             f"iterative method {iterative_method} not found.\n supported methods are:\n {ITERATIVE_METHODS}"
#         )

#     # 3. Call iterative solver with neural network initial guess
#     x, info = solver_fn(A, b, x0=x0, atol=ATOL, **kwargs)

#     # 4. Check convergence
#     if info != 0:
#         logger.warning(f"Iterative solver did not converge! Info: {info}")

#     return x


# # Update solve_linear to include the new solver option
# def solve_linear(
#     A,
#     b,
#     solver_type="direct",
#     neural_solver=None,
#     eps=None,
#     iterative_method=None,
#     symmetry=False,
#     **kwargs,
# ):
#     """Master function to call different solvers"""

#     if solver_type == "direct":
#         return _solve_direct(A, b, symmetry=symmetry)
#     elif solver_type == "iterative":
#         return _solve_iterative(A, b, iterative_method=iterative_method, **kwargs)
#     elif solver_type == "iterative_nn":
#         return _solve_iterative_with_nn_init(
#             A, b, neural_solver, eps, iterative_method=iterative_method, **kwargs
#         )
#     else:
#         raise ValueError(f"Solver type {solver_type} not supported")


# # def solve_linear(A, b, iterative_method=False, symmetry=False, **kwargs):
# #     """ Master function to call the others """

# #     if iterative_method and iterative_method is not None:
# #         # if iterative solver string is supplied, use that method
# #         return _solve_iterative(A, b, iterative_method=iterative_method, **kwargs)
# #     elif iterative_method and iterative_method is None:
# #         # if iterative_method is supplied as None, use the default
# #         return _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD)
# #     else:
# #         # otherwise, use a direct solver
# #         return _solve_direct(A, b, symmetry=symmetry)


# def _solve_direct(A, b, symmetry=False):
#     """Direct solver"""

#     if HAS_MKL:
#         # prefered method using MKL. Much faster (on Mac at least)
#         if symmetry:
#             mtype = 6
#         else:
#             mtype = 13
#         pSolve = pardisoSolver(A, mtype=mtype)
#         pSolve.factor()
#         x = pSolve.solve(b)
#         pSolve.clear()
#         return x
#     else:
#         # scipy solver.
#         return spl.spsolve(A, b)


# def _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD, **kwargs):
#     """Iterative solver"""

#     # error checking on the method name (https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
#     try:
#         solver_fn = ITERATIVE_METHODS[iterative_method]
#     except:
#         raise ValueError(
#             "iterative method {} not found.\n supported methods are:\n {}".format(
#                 iterative_method, ITERATIVE_METHODS
#             )
#         )

#     # call the solver using scipy's API
#     x, info = solver_fn(A, b, atol=ATOL, **kwargs)
#     return x


# def _solve_cuda(A, b, **kwargs):
#     """You could put some other solver here if you're feeling adventurous"""
#     raise NotImplementedError("Please implement something fast and exciting here!")


# # ---------------------- Sparse Solver Copied from Ceviche ----------------------

# # Custom PyTorch sparse solver exploiting a CuPy backend
# # See https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html
# # class SparseSolveCupy(torch.autograd.Function):
# #     @staticmethod
# #     def forward(ctx, A, b):
# #         # Sanity check
# #         if A.ndim != 2 or (A.shape[0] != A.shape[1]):
# #             raise ValueError("A should be a square 2D matrix.")
# #         # Transfer data to CuPy
# #         A_cp = coo_torch2cupy(A)
# #         b_cp = cp.asarray(b.data)
# #         # Solver the sparse system
# #         ctx.factorisedsolver = None
# #         if (b.ndim == 1) or (b.shape[1] == 1):
# #             # cp.sparse.linalg.spsolve only works if b is a vector but is fully on GPU
# #             x_cp = spsolve(A_cp, b_cp)
# #         else:
# #             # Make use of a factorisation (only the solver is then on the GPU)
# #             # We store it in ctx to reuse it in the backward pass
# #             ctx.factorisedsolver = factorized(A_cp)
# #             x_cp = ctx.factorisedsolver(b_cp)
# #         # Transfer (dense) result back to PyTorch
# #         x = torch.as_tensor(x_cp, device=b.device)
# #         if A.requires_grad or b.requires_grad:
# #             # Not sure if the following is needed / helpful
# #             x.requires_grad = True
# #         else:
# #             # Free up memory
# #             ctx.factorisedsolver = None
# #         # Save context for backward pass
# #         ctx.save_for_backward(A, b, x)
# #         return x

# #     @staticmethod
# #     def backward(ctx, grad):
# #         # Recover context
# #         A, b, x = ctx.saved_tensors
# #         # Compute gradient with respect to b
# #         if ctx.factorisedsolver is None:
# #             gradb = SparseSolve.apply(A.t(), grad)
# #         else:
# #             # Re-use factorised solver from forward pass
# #             grad_cp = cp.asarray(grad.data)
# #             gradb_cp = ctx.factorisedsolver(grad_cp, trans="T")
# #             gradb = torch.as_tensor(gradb_cp, device=b.device)
# #         # The gradient with respect to the (dense) matrix A would be something like
# #         # -gradb @ x.T but we are only interested in the gradient with respect to
# #         # the (non-zero) values of A
# #         gradAidx = A.indices()
# #         mgradbselect = -gradb.index_select(0, gradAidx[0, :])
# #         xselect = x.index_select(0, gradAidx[1, :])
# #         mgbx = mgradbselect * xselect
# #         if x.dim() == 1:
# #             gradAvals = mgbx
# #         else:
# #             gradAvals = torch.sum(mgbx, dim=1)
# #         gradAs = torch.sparse_coo_tensor(gradAidx, gradAvals, A.shape).to_sparse_csr()
# #         # gradAs = torch.sparse_csr_tensor(gradAs)
# #         return gradAs, gradb


# # sparse_solve_cupy = SparseSolveCupy.apply


# class SparseSolveTorchFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         entries_a,
#         indices_a,
#         eps_matrix: Tensor,  # for Ez, this is the diagonal of A, for Hz, this should be a sparse matrix
#         omega: float,
#         b: np.ndarray | Tensor,
#         solver_instance,
#         slice_name,
#         mode,
#         temp,
#         Pl,
#         Pr,
#         neural_solver,
#         numerical_solver,
#         shape,
#         use_autodiff=False,
#         eps: Tensor | None = None,  # 2D array of eps_r, can be used in neural solver
#         pol: str = "Ez",  # Ez or Hz
#         simulation_id: int = 0,
#     ):
#         ### entries_a: values of the sparse matrix A
#         ### indices_a: row/column indices of the sparse matrix A
#         ### eps_diag: For Ez diagonal of A, e.g., -omega**2 * epr_0 * eps_r
#         ctx.use_autodiff = use_autodiff
#         if use_autodiff:
#             assert (
#                 numerical_solver == "none"
#             ), f"numerical_solver {numerical_solver} is not supported when use_autodiff is True"
#         Jz = b / (1j * omega)
#         if isinstance(Jz, np.ndarray) and neural_solver is not None:
#             if neural_solver["fwd_solver"] is not None:
#                 Jz = torch.from_numpy(Jz).to(neural_solver["fwd_solver"].device)
#             else:
#                 pass  # we will not use Jz
#         if Jz.dtype == torch.complex128:
#             Jz = Jz.to(torch.complex64)
#         if isinstance(b, Tensor):
#             b = b.cpu().numpy()
#         N = np.prod(shape)
#         A = make_sparse(entries_a, indices_a, (N, N))
#         # epsilon = torch.tensor([EPSILON_0,], dtype=eps_diag.dtype, device=eps_diag.device)
#         # omega = torch.tensor([omega,], dtype=eps_diag.dtype, device=eps_diag.device)
#         # eps = (eps_diag / (-EPSILON_0 * omega**2)).reshape(shape) # here there is an assumption that the eps_vec is a square matrix
#         Jz = Jz.reshape(shape)

#         if Pl is not None and Pr is not None:
#             A = Pl @ A @ Pr
#             b = Pl @ b
#             symmetry = True
#         else:
#             symmetry = False
#         # with TimerCtx() as t:
#         if numerical_solver == "solve_direct":
#             x = solve_linear(A, b, symmetry=symmetry)
#             # x = solve_linear(A, b, iterative_method="lgmres", symmetry=symmetry, rtol=1e-2)
#         # print(f"my solve time (symmetry={symmetry}): {t.interval}")
#         elif numerical_solver == "none":
#             assert neural_solver is not None
#             # print("we are now using pure neural solver", flush=True)
#             fwd_solver = neural_solver["fwd_solver"]
#             x = fwd_solver(
#                 eps.unsqueeze(0),
#                 Jz.unsqueeze(0),
#             )
#             if isinstance(x, tuple):
#                 x = x[0]
#             x = x.permute(0, 2, 3, 1).contiguous()
#             x = torch.view_as_complex(x)
#             x = x.flatten()
#         elif numerical_solver == "solve_iterative":
#             if neural_solver is not None:
#                 fwd_solver = neural_solver["fwd_solver"]
#                 solver_type = "iterative_nn"
#             else:
#                 fwd_solver = None
#                 solver_type = "iterative"
#             x = solve_linear(
#                 A,
#                 b,
#                 solver_type=solver_type,
#                 neural_solver=fwd_solver,
#                 eps=eps,
#                 iterative_method="bicgstab"
#                 if symmetry
#                 else "lgmres",  # or other methods
#                 symmetry=symmetry,
#                 maxiter=1000,
#                 rtol=1e-2,
#             )
#         else:
#             raise ValueError(f"numerical_solver {numerical_solver} not supported")
#         if Pl is not None and Pr is not None:
#             x = Pr @ x
#         if not isinstance(x, torch.Tensor):
#             x = torch.from_numpy(x).to(torch.complex128).to(eps_matrix.device)
#         ctx.entries_a = entries_a
#         ctx.indices_a = np.flip(indices_a, axis=0)
#         ctx.save_for_backward(x, eps_matrix)
#         ctx.solver_instance = solver_instance
#         ctx.slice_name = slice_name
#         ctx.mode = mode
#         ctx.temp = temp
#         ctx.Pl = Pl
#         ctx.Pr = Pr
#         ctx.shape = shape
#         ctx.pol = pol
#         ctx.adj_solver = (
#             neural_solver["adj_solver"] if neural_solver is not None else None
#         )
#         ctx.numerical_solver = numerical_solver
#         ctx.eps = eps
#         ctx.omega = omega
#         ctx.simulation_id = simulation_id
#         return x

#     @staticmethod
#     def backward(ctx, grad):
#         if ctx.use_autodiff:
#             return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
#         else:
#             adj_solver = ctx.adj_solver
#             numerical_solver = ctx.numerical_solver
#             eps = ctx.eps
#             shape = ctx.shape
#             x, eps_matrix = ctx.saved_tensors
#             entries_a = ctx.entries_a
#             indices_a = ctx.indices_a
#             solver_instance = ctx.solver_instance
#             slice_name = ctx.slice_name
#             mode = ctx.mode
#             temp = ctx.temp
#             N = np.prod(shape)
#             Pl, Pr = ctx.Pl, ctx.Pr
#             omega = ctx.omega
#             simulation_id = ctx.simulation_id
#             A_t = make_sparse(entries_a, indices_a, (N, N))
#             grad = grad.cpu().numpy().astype(np.complex128)
#             adj_src = grad.conj()
#             if (slice_name != "Norm" and mode != "Norm" and temp != "Norm") or (
#                 slice_name != "adj" and mode != "adj" and temp != "adj"
#             ):
#                 solver_instance.adj_src[(slice_name, mode, temp)] = (
#                     torch.from_numpy(adj_src).to(torch.complex128).to(eps_matrix.device)
#                 )
#             ## this adj_src = "-v" in ceviche
#             # print_stat(adj_src, "my adjoint source")
#             # print(f"my adjoint A_t", A_t)
#             if Pl is not None and Pr is not None:
#                 A_t = Pr.T @ A_t @ Pl.T
#                 adj_src = Pr.T @ adj_src
#                 symmetry = True
#             else:
#                 symmetry = False
#             if numerical_solver == "solve_direct":
#                 adj = solve_linear(A_t, adj_src, symmetry=symmetry)
#             elif numerical_solver == "none":
#                 assert adj_solver is not None
#                 # print("we are now using pure neural solver for adjoint", flush=True)
#                 if isinstance(adj_src, np.ndarray):
#                     adj_src = (
#                         torch.from_numpy(adj_src)
#                         .to(torch.complex64)
#                         .to(eps_matrix.device)
#                     )
#                 adj_src = adj_src.reshape(shape)
#                 adj = adj_solver(
#                     eps.unsqueeze(0),
#                     adj_src.unsqueeze(0),
#                 )
#                 if isinstance(adj, tuple):
#                     adj = adj[-1]
#                 adj = adj.permute(0, 2, 3, 1).contiguous()
#                 adj = torch.view_as_complex(adj)
#                 adj = adj.flatten()
#                 adj = adj / 1e23
#             elif numerical_solver == "solve_iterative":
#                 adj = solve_linear(
#                     A_t,
#                     adj_src,
#                     solver_type="iterative_nn",
#                     neural_solver=adj_solver,
#                     eps=eps,
#                     iterative_method="lgmres",  # or other methods
#                     symmetry=symmetry,
#                     maxiter=1000,
#                     rtol=1e-2,
#                 )
#             else:
#                 raise ValueError(f"numerical_solver {numerical_solver} not supported")
#             if Pl is not None and Pr is not None:
#                 adj = Pl.T @ adj
#             if not isinstance(adj, torch.Tensor):
#                 adj = torch.from_numpy(adj).to(torch.complex128).to(eps_matrix.device)
#             solver_instance.adj_field[(slice_name, mode, temp)] = adj

#             if ctx.pol == "Ez":
#                 grad_epsilon = -(adj.mul_(x).real.to(eps_matrix.device))
#             elif ctx.pol == "Hz":
#                 indices = eps_matrix.indices()
#                 grad_epsilon = -(
#                     adj[indices[0]].mul_(x[indices[1]]).real.to(eps_matrix.device)
#                 )
#                 grad_epsilon = torch.sparse_coo_tensor(
#                     indices, grad_epsilon, eps_matrix.shape
#                 )
#             else:
#                 raise ValueError(f"pol {ctx.pol} not supported")
#             ## this grad_epsilon = adj * x in ceviche

#             # print(f"my grad eps", grad_epsilon)

#             # if b.requires_grad:
#             #     grad_b = solve_linear(A_t, grad)
#             #     grad_b = torch.from_numpy(grad_b).to(torch.complex128).to(b.device)
#             # else:
#             if (slice_name != "Norm" and mode != "Norm" and temp != "Norm") or (
#                 slice_name != "adj" and mode != "adj" and temp != "adj"
#             ):
#                 solver_instance.gradient[(slice_name, mode, temp)] = (
#                     grad_epsilon.to(torch.float32).detach()
#                 )
#             grad_b = None
#             return (
#                 None,
#                 None,
#                 grad_epsilon,
#                 None,
#                 grad_b,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#             )


# class SparseSolveTorch(torch.nn.Module):
#     def __init__(self, shape, neural_solver, numerical_solver, use_autodiff=False):
#         super(SparseSolveTorch, self).__init__()
#         self.shape = shape
#         self.adj_src = {}  # now the adj_src is a dictionary in which the key is (port_name, mode) with same wl, different wl have different simulation objects
#         self.adj_field = {}
#         self.gradient = {}
#         self.neural_solver = neural_solver
#         self.numerical_solver = numerical_solver
#         self.use_autodiff = use_autodiff

#     def set_shape(self, shape):
#         self.shape = shape

#     def forward(
#         self,
#         entries_a,
#         indices_a,
#         eps_matrix: Tensor,
#         omega: float,
#         b: np.ndarray | Tensor,
#         slice_name,
#         mode,
#         temp,
#         Pl=None,
#         Pr=None,
#         eps: Tensor | None = None,
#         pol: str = "Ez",
#         simulation_id: int = 0,
#     ):
#         x = SparseSolveTorchFunction.apply(
#             entries_a,
#             indices_a,
#             eps_matrix,
#             omega,
#             b,
#             self,
#             slice_name,
#             mode,
#             temp, 
#             Pl,
#             Pr,
#             self.neural_solver,
#             self.numerical_solver,
#             self.shape,
#             self.use_autodiff,
#             eps,
#             pol,
#             simulation_id
#         )
#         return x


# sparse_solve_torch = SparseSolveTorchFunction.apply





# this is the solver in hyper 
"""
Date: 2024-10-10 19:50:23
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-03-07 00:36:41
FilePath: /MAPS/core/fdfd/solver.py
"""

import cupy as cp
import numpy as np
import scipy.sparse.linalg as spl
import torch
from pyMKL import pardisoSolver
from pyutils.general import TimerCtx, logger
from torch import Tensor

from ...thirdparty.ceviche.constants import *
from ...thirdparty.ceviche.utils import make_sparse
from ...core.utils import print_stat
from .cudss_spsolve import spsolve_cudss
import time
import gc
import sys
try:
    from pyMKL import pardisoSolver
    # from .pardiso_solver import pardisoSolver

    HAS_MKL = True
    # print('using MKL for direct solvers')
except:
    HAS_MKL = False

ENABLE_TIMER = False


def coo_torch2cupy(A):
    A = A.data.coalesce()
    Avals_cp = cp.asarray(A.values())
    Aidx_cp = cp.asarray(A.indices())
    # return cp.sparse.csr_matrix((Avals_cp, Aidx_cp), shape=A.shape)
    return cp.sparse.coo_matrix((Avals_cp, Aidx_cp))


# ---------------------- Sparse Solver Copied from Ceviche ----------------------
DEFAULT_ITERATIVE_METHOD = "bicg"

# dict of iterative methods supported (name: function)
ITERATIVE_METHODS = {
    "bicg": spl.bicg,
    "bicgstab": spl.bicgstab,
    "cg": spl.cg,
    "cgs": spl.cgs,
    "gmres": spl.gmres,
    "lgmres": spl.lgmres,
    "qmr": spl.qmr,
    "gcrotmk": spl.gcrotmk,
}

# convergence tolerance for iterative solvers.
ATOL = 1e-8

""" ========================== SOLVER FUNCTIONS ========================== """


def _solve_iterative_with_nn_init(
    A, b, neural_solver, eps, iterative_method=DEFAULT_ITERATIVE_METHOD, **kwargs
):
    """Iterative solver with neural network initial guess"""

    # 1. Get initial guess from neural network
    if neural_solver is not None:
        # Convert b to a PyTorch tensor if it's a numpy array
        if isinstance(b, np.ndarray):
            b_torch = (
                torch.from_numpy(b).to(eps.device).to(torch.complex64)
            )  # Ensure the tensor is on the same device as eps
        else:
            b_torch = b
        x0 = neural_solver(
            eps.unsqueeze(0),  # Add batch dimension
            b_torch.reshape(eps.shape).unsqueeze(
                0
            ),  # Reshape source term to match eps shape
        )
        if isinstance(x0, tuple):
            x0 = x0[-1]  # Get final prediction if model has error correction
        # Reshape NN output to match linear system size
        x0 = x0.permute(0, 2, 3, 1).contiguous()
        x0 = torch.view_as_complex(x0)
        x0 = x0.squeeze(0).flatten()
        x0 = x0.detach().cpu().numpy()
        print("now we have the initial guess", flush=True)
    else:
        x0 = None  # Use default zero initial guess

    # 2. Get the solver function
    try:
        solver_fn = ITERATIVE_METHODS[iterative_method]
    except:
        raise ValueError(
            f"iterative method {iterative_method} not found.\n supported methods are:\n {ITERATIVE_METHODS}"
        )

    # 3. Call iterative solver with neural network initial guess
    x, info = solver_fn(A, b, x0=x0, atol=ATOL, **kwargs)

    # 4. Check convergence
    if info != 0:
        logger.warning(f"Iterative solver did not converge! Info: {info}")

    return x


# Update solve_linear to include the new solver option
def solve_linear(
    A,
    b,
    solver_type="direct",
    neural_solver=None,
    eps=None,
    iterative_method=None,
    symmetry=False,
    clear: bool = True,
    double: bool = True,
    **kwargs,
):
    """Master function to call different solvers"""

    if solver_type == "direct":
        return _solve_direct(A, b, symmetry=symmetry, clear=clear, double=double)
    elif solver_type == "iterative":
        return _solve_iterative(A, b, iterative_method=iterative_method, **kwargs)
    elif solver_type == "iterative_nn":
        return _solve_iterative_with_nn_init(
            A, b, neural_solver, eps, iterative_method=iterative_method, **kwargs
        )
    else:
        raise ValueError(f"Solver type {solver_type} not supported")


# def solve_linear(A, b, iterative_method=False, symmetry=False, **kwargs):
#     """ Master function to call the others """

#     if iterative_method and iterative_method is not None:
#         # if iterative solver string is supplied, use that method
#         return _solve_iterative(A, b, iterative_method=iterative_method, **kwargs)
#     elif iterative_method and iterative_method is None:
#         # if iterative_method is supplied as None, use the default
#         return _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD)
#     else:
#         # otherwise, use a direct solver
#         return _solve_direct(A, b, symmetry=symmetry)


def _solve_direct(A, b, symmetry=False, clear: bool = True, double: bool = True):
    """Direct solver"""

    if HAS_MKL:
        # prefered method using MKL. Much faster (on Mac at least)
        if symmetry:
            mtype = 6
        else:
            mtype = 13
        # pSolve = pardisoSolver(A, mtype=mtype, double=double)
        pSolve = pardisoSolver(A, mtype=mtype)
        pSolve.factor()
        x = pSolve.solve(b)
        if clear:
            pSolve.clear()
            return x, None
        else:
            return x, pSolve
    else:
        # scipy solver.
        return spl.spsolve(A, b)


def _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD, **kwargs):
    """Iterative solver"""

    # error checking on the method name (https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
    try:
        solver_fn = ITERATIVE_METHODS[iterative_method]
    except:
        raise ValueError(
            "iterative method {} not found.\n supported methods are:\n {}".format(
                iterative_method, ITERATIVE_METHODS
            )
        )

    # call the solver using scipy's API
    x, info = solver_fn(A, b, atol=ATOL, **kwargs)
    return x


def _solve_cuda(A, b, **kwargs):
    """You could put some other solver here if you're feeling adventurous"""
    raise NotImplementedError("Please implement something fast and exciting here!")


# ---------------------- Sparse Solver Copied from Ceviche ----------------------

# Custom PyTorch sparse solver exploiting a CuPy backend
# See https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html
# class SparseSolveCupy(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, A, b):
#         # Sanity check
#         if A.ndim != 2 or (A.shape[0] != A.shape[1]):
#             raise ValueError("A should be a square 2D matrix.")
#         # Transfer data to CuPy
#         A_cp = coo_torch2cupy(A)
#         b_cp = cp.asarray(b.data)
#         # Solver the sparse system
#         ctx.factorisedsolver = None
#         if (b.ndim == 1) or (b.shape[1] == 1):
#             # cp.sparse.linalg.spsolve only works if b is a vector but is fully on GPU
#             x_cp = spsolve(A_cp, b_cp)
#         else:
#             # Make use of a factorisation (only the solver is then on the GPU)
#             # We store it in ctx to reuse it in the backward pass
#             ctx.factorisedsolver = factorized(A_cp)
#             x_cp = ctx.factorisedsolver(b_cp)
#         # Transfer (dense) result back to PyTorch
#         x = torch.as_tensor(x_cp, device=b.device)
#         if A.requires_grad or b.requires_grad:
#             # Not sure if the following is needed / helpful
#             x.requires_grad = True
#         else:
#             # Free up memory
#             ctx.factorisedsolver = None
#         # Save context for backward pass
#         ctx.save_for_backward(A, b, x)
#         return x

#     @staticmethod
#     def backward(ctx, grad):
#         # Recover context
#         A, b, x = ctx.saved_tensors
#         # Compute gradient with respect to b
#         if ctx.factorisedsolver is None:
#             gradb = SparseSolve.apply(A.t(), grad)
#         else:
#             # Re-use factorised solver from forward pass
#             grad_cp = cp.asarray(grad.data)
#             gradb_cp = ctx.factorisedsolver(grad_cp, trans="T")
#             gradb = torch.as_tensor(gradb_cp, device=b.device)
#         # The gradient with respect to the (dense) matrix A would be something like
#         # -gradb @ x.T but we are only interested in the gradient with respect to
#         # the (non-zero) values of A
#         gradAidx = A.indices()
#         mgradbselect = -gradb.index_select(0, gradAidx[0, :])
#         xselect = x.index_select(0, gradAidx[1, :])
#         mgbx = mgradbselect * xselect
#         if x.dim() == 1:
#             gradAvals = mgbx
#         else:
#             gradAvals = torch.sum(mgbx, dim=1)
#         gradAs = torch.sparse_coo_tensor(gradAidx, gradAvals, A.shape).to_sparse_csr()
#         # gradAs = torch.sparse_csr_tensor(gradAs)
#         return gradAs, gradb


# sparse_solve_cupy = SparseSolveCupy.apply
SCALE = 1e15


class SparseSolveTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        entries_a,
        indices_a,
        eps_matrix: Tensor,  # for Ez, this is the diagonal of A, for Hz, this should be a sparse matrix
        omega: float,
        b: np.ndarray | Tensor,
        solver_instance,
        slice_name,
        mode,
        temp,
        Pl,
        Pr,
        neural_solver,
        numerical_solver,
        shape,
        use_autodiff=False,
        eps: Tensor | None = None,  # 2D array of eps_r, can be used in neural solver
        pol: str = "Ez",  # Ez or Hz
        double: bool = False,
        _solver_cache: dict | None = None,
        simulation_id: int = 0,
    ):
        ### entries_a: values of the sparse matrix A
        ### indices_a: row/column indices of the sparse matrix A
        ### eps_diag: For Ez diagonal of A, e.g., -omega**2 * epr_0 * eps_r
        ctx.use_autodiff = use_autodiff
        if use_autodiff:
            assert numerical_solver == "none", (
                f"numerical_solver {numerical_solver} is not supported when use_autodiff is True"
            )
        Jz = b / (1j * omega)
        if isinstance(Jz, np.ndarray) and neural_solver is not None:
            if neural_solver["fwd_solver"] is not None:
                Jz = torch.from_numpy(Jz).to(neural_solver["fwd_solver"].device)
            else:
                pass  # we will not use Jz
        if Jz.dtype == torch.complex128:
            Jz = Jz.to(torch.complex64)
        if isinstance(b, Tensor):
            b = b.cpu().numpy()
        N = np.prod(shape)
        A = make_sparse(entries_a, indices_a, (N, N))
        # epsilon = torch.tensor([EPSILON_0,], dtype=eps_diag.dtype, device=eps_diag.device)
        # omega = torch.tensor([omega,], dtype=eps_diag.dtype, device=eps_diag.device)
        # eps = (eps_diag / (-EPSILON_0 * omega**2)).reshape(shape) # here there is an assumption that the eps_vec is a square matrix
        Jz = Jz.reshape(shape)

        if Pl is not None and Pr is not None:
            A = Pl @ A @ Pr
            b = Pl @ b
            symmetry = True
            ctx.precond_A = A
        else:
            symmetry = False

        if numerical_solver == "solve_direct":
            with TimerCtx(enable=ENABLE_TIMER, desc="forward"):
                pSolve = (
                    _solver_cache.get(f"pSolve_fwd_{simulation_id}", None)
                    if _solver_cache is not None
                    else None
                )
                ctx.pSolve = None
                if not double:
                    b = (b / SCALE).astype(np.complex64)

                if pSolve is not None:
                    x = _solver_cache[f"pSolve_fwd_{simulation_id}"].solve(b)
                else:
                    if not double:
                        A = (A / SCALE).astype(np.complex64)

                        # print("max A and b", np.max(A), np.max(b))
                    ### if not cache mode and not symmetric A, we cannot reuse anything, just clear it.
                    x, pSolve = solve_linear(
                        A,
                        b,
                        symmetry=symmetry,
                        clear=not _solver_cache and not symmetry,
                        double=double,
                    )
                    if _solver_cache is not None:
                        _solver_cache[f"pSolve_fwd_{simulation_id}"] = pSolve
                if symmetry:  # only when A is symmetric after precondiction, we can reuse pSolve for adjoint
                    ctx.pSolve = pSolve
                # print("forward x", np.max(x))

                # x = solve_linear(A, b, iterative_method="lgmres", symmetry=symmetry, rtol=1e-2)
            # print(f"my solve time (symmetry={symmetry}): {t.interval}")
        elif numerical_solver == "solve_direct_gpu":
            assert symmetry and not double
            with TimerCtx(enable=ENABLE_TIMER, desc="forward"):
                if not double:
                    b = (b / SCALE).astype(np.complex64)
                    A = (A / SCALE).astype(np.complex64)

                with torch.cuda.stream(torch.cuda.default_stream()):
                    A = A.tocoo().tocsr().sorted_indices()
                    x = spsolve_cudss(
                        A, b, device=eps_matrix.device, mtype=1 if symmetry else 0
                    )
                torch.cuda.synchronize()

                # print("forward x", np.max(x))
        elif numerical_solver == "none":
            assert neural_solver is not None
            # print("we are now using pure neural solver", flush=True)
            fwd_solver = neural_solver["fwd_solver"]
            x = fwd_solver(
                eps.unsqueeze(0),
                Jz.unsqueeze(0),
            )
            if isinstance(x, tuple):
                x = x[0]
            x = x.permute(0, 2, 3, 1).contiguous()
            x = torch.view_as_complex(x)
            x = x.flatten()
        elif numerical_solver == "solve_iterative":
            if neural_solver is not None:
                fwd_solver = neural_solver["fwd_solver"]
                solver_type = "iterative_nn"
            else:
                fwd_solver = None
                solver_type = "iterative"
            x = solve_linear(
                A,
                b,
                solver_type=solver_type,
                neural_solver=fwd_solver,
                eps=eps,
                iterative_method="bicgstab"
                if symmetry
                else "lgmres",  # or other methods
                symmetry=symmetry,
                maxiter=1000,
                rtol=1e-2,
            )
        else:
            raise ValueError(f"numerical_solver {numerical_solver} not supported")
        if Pl is not None and Pr is not None:
            x = Pr @ x
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).to(torch.complex128).to(eps_matrix.device)
        ctx.entries_a = entries_a
        ctx.indices_a = np.flip(indices_a, axis=0)
        ctx.save_for_backward(x, eps_matrix)
        ctx.solver_instance = solver_instance
        ctx.slice_name = slice_name
        ctx.mode = mode
        ctx.temp = temp
        ctx.Pl = Pl
        ctx.Pr = Pr
        ctx.shape = shape
        ctx.pol = pol
        ctx.adj_solver = (
            neural_solver["adj_solver"] if neural_solver is not None else None
        )
        ctx.numerical_solver = numerical_solver
        ctx.eps = eps
        ctx.omega = omega
        ctx.double = double
        ctx._solver_cache = _solver_cache
        ctx.simulation_id = simulation_id
        return x

    @staticmethod
    def backward(ctx, grad):
        if ctx.use_autodiff:
            return tuple([None] * 20)
        else:
            adj_solver = ctx.adj_solver
            numerical_solver = ctx.numerical_solver
            eps = ctx.eps
            shape = ctx.shape
            x, eps_matrix = ctx.saved_tensors
            entries_a = ctx.entries_a
            indices_a = ctx.indices_a
            solver_instance = ctx.solver_instance
            slice_name = ctx.slice_name
            mode = ctx.mode
            temp = ctx.temp
            N = np.prod(shape)
            Pl, Pr = ctx.Pl, ctx.Pr
            simulation_id = ctx.simulation_id

            grad = grad.cpu().numpy().astype(np.complex128)
            adj_src = grad.conj()
            if (slice_name != "Norm" and mode != "Norm" and temp != "Norm") or (
                slice_name != "adj" and mode != "adj" and temp != "adj"
            ):
                solver_instance.adj_src[(slice_name, mode, temp)] = (
                    torch.from_numpy(adj_src).to(torch.complex128).to(eps_matrix.device)
                )
            ## this adj_src = "-v" in ceviche
            # print_stat(adj_src, "my adjoint source")
            # print(f"my adjoint A_t", A_t)
            if Pl is not None and Pr is not None:
                # A_t = Pr.T @ A_t @ Pl.T
                A_t = ctx.precond_A
                adj_src = Pr.T @ adj_src
                symmetry = True
            else:
                A_t = make_sparse(entries_a, indices_a, (N, N))
                symmetry = False
            if numerical_solver == "solve_direct":
                with TimerCtx(enable=ENABLE_TIMER, desc="adjoint"):
                    if not ctx.double:
                        adj_src = (adj_src / SCALE).astype(np.complex64)

                    if ctx.pSolve is None:
                        ## no fwd solver to reuse, means A is not symmtric
                        ## if we find cached adj solver, we reuse it
                        if (
                            ctx._solver_cache is not None
                            and ctx._solver_cache[f"pSolve_adj_{simulation_id}"] is not None
                        ):
                            # adj = ctx.pSolve.solve(adj_src)
                            adj = ctx._solver_cache[f"pSolve_adj_{simulation_id}"].solve(
                                adj_src
                            )
                        else:
                            ## nothing to reuse, we need to solve it
                            if not ctx.double:
                                A_t = (A_t / SCALE).astype(np.complex64)
                            adj, pSolve = solve_linear(
                                A_t,
                                adj_src,
                                symmetry=symmetry,
                                clear=not ctx._solver_cache,
                                double=ctx.double,
                            )
                            # if cache mode, we need to cache this adj solver as well
                            if ctx._solver_cache is not None:
                                ctx._solver_cache[f"pSolve_adj_{simulation_id}"] = pSolve
                    else:
                        ## we reuse fwd solver
                        adj = ctx.pSolve.solve(adj_src)
                        ctx.pSolve.clear()
                        ctx.pSolve = None
                        # gc.collect()
                    # print("backward adj", adj[100])

            elif numerical_solver == "solve_direct_gpu":
                assert symmetry and not ctx.double
                with TimerCtx(enable=ENABLE_TIMER, desc="adjoint"):
                    if not ctx.double:
                        adj_src = (adj_src / SCALE).astype(np.complex64)
                        A_t = (A_t / SCALE).astype(np.complex64)

                    with torch.cuda.stream(torch.cuda.default_stream()):
                        A_t = A_t.tocoo().tocsr().sorted_indices()
                        adj = spsolve_cudss(
                            A_t,
                            adj_src,
                            device=eps_matrix.device,
                            mtype=1 if symmetry else 0,
                        )
                    torch.cuda.synchronize()

            elif numerical_solver == "none":
                assert adj_solver is not None
                # print("we are now using pure neural solver for adjoint", flush=True)
                if isinstance(adj_src, np.ndarray):
                    adj_src = (
                        torch.from_numpy(adj_src)
                        .to(torch.complex64)
                        .to(eps_matrix.device)
                    )
                adj_src = adj_src.reshape(shape)
                adj = adj_solver(
                    eps.unsqueeze(0),
                    adj_src.unsqueeze(0),
                )
                if isinstance(adj, tuple):
                    adj = adj[-1]
                adj = adj.permute(0, 2, 3, 1).contiguous()
                adj = torch.view_as_complex(adj)
                adj = adj.flatten()
                adj = adj / 1e23
            elif numerical_solver == "solve_iterative":
                adj = solve_linear(
                    A_t,
                    adj_src,
                    solver_type="iterative_nn",
                    neural_solver=adj_solver,
                    eps=eps,
                    iterative_method="lgmres",  # or other methods
                    symmetry=symmetry,
                    maxiter=1000,
                    rtol=1e-2,
                )
            else:
                raise ValueError(f"numerical_solver {numerical_solver} not supported")
            if Pl is not None and Pr is not None:
                adj = Pl.T @ adj
            if not isinstance(adj, torch.Tensor):
                adj = torch.from_numpy(adj).to(torch.complex128).to(eps_matrix.device)
            solver_instance.adj_field[(slice_name, mode, temp)] = adj

            if ctx.pol == "Ez":
                grad_epsilon = -(adj.mul_(x).real.to(eps_matrix.device))
            elif ctx.pol == "Hz":
                indices = eps_matrix.indices()
                grad_epsilon = -(
                    adj[indices[0]].mul_(x[indices[1]]).real.to(eps_matrix.device)
                )
                grad_epsilon = torch.sparse_coo_tensor(
                    indices, grad_epsilon, eps_matrix.shape
                )
            else:
                raise ValueError(f"pol {ctx.pol} not supported")
            ## this grad_epsilon = adj * x in ceviche

            # print(f"my grad eps", grad_epsilon)

            # if b.requires_grad:
            #     grad_b = solve_linear(A_t, grad)
            #     grad_b = torch.from_numpy(grad_b).to(torch.complex128).to(b.device)
            # else:
            if (slice_name != "Norm" and mode != "Norm" and temp != "Norm") or (
                slice_name != "adj" and mode != "adj" and temp != "adj"
            ):
                solver_instance.gradient[(slice_name, mode, temp)] = grad_epsilon.to(
                    torch.float32
                ).detach()
            grad_b = None
            return (
                None,
                None,
                grad_epsilon,
                None,
                grad_b,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )


class SparseSolveTorch(torch.nn.Module):
    def __init__(self, shape, neural_solver, numerical_solver, use_autodiff=False):
        super(SparseSolveTorch, self).__init__()
        self.shape = shape
        self.adj_src = {}  # now the adj_src is a dictionary in which the key is (port_name, mode) with same wl, different wl have different simulation objects
        self.adj_field = {}
        self.gradient = {}
        self.neural_solver = neural_solver
        self.numerical_solver = numerical_solver
        self.use_autodiff = use_autodiff
        self._solver_cache = {"pSolve_fwd": None, "pSolve_adj": None}

        self.set_cache_mode(False)

    def set_shape(self, shape):
        self.shape = shape

    def set_cache_mode(self, mode: bool):
        self._cache_mode = mode
        if not mode:
            self.clear_solver_cache()

    def clear_solver_cache(self):
        for key, value in self._solver_cache.items():
            if value is not None:
                value.clear()
                self._solver_cache[key] = None
        # if self._solver_cache["pSolve_fwd"] is not None:
        #     self._solver_cache["pSolve_fwd"].clear()
        #     self._solver_cache["pSolve_fwd"] = None
        # if self._solver_cache["pSolve_adj"] is not None:
        #     self._solver_cache["pSolve_adj"].clear()
        #     self._solver_cache["pSolve_adj"] = None

    def forward(
        self,
        entries_a,
        indices_a,
        eps_matrix: Tensor,
        omega: float,
        b: np.ndarray | Tensor,
        slice_name,
        mode,
        temp,
        Pl=None,
        Pr=None,
        eps: Tensor | None = None,
        pol: str = "Ez",
        double: bool = True,
        simulation_id: int = 0,
    ):
        x = SparseSolveTorchFunction.apply(
            entries_a,
            indices_a,
            eps_matrix,
            omega,
            b,
            self,
            slice_name,
            mode,
            temp,
            Pl,
            Pr,
            self.neural_solver,
            self.numerical_solver,
            self.shape,
            self.use_autodiff,
            eps,
            pol,
            double,
            self._solver_cache if self._cache_mode else None,
            simulation_id,
        )
        return x

sparse_solve_torch = SparseSolveTorchFunction.apply
