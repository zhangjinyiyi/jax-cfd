# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for computing and applying pressure."""

from typing import Callable, Optional, Sequence

import jax.numpy as jnp
import jax.scipy.sparse.linalg

from jax_cfd.base import array_utils
from jax_cfd.base import fast_diagonalization
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import grids

Array = grids.Array
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector

# Specifying the full signatures of Callable would get somewhat onerous
# pylint: disable=g-bare-generic


def solve_cg(v: Sequence[GridArray],
             q0: Optional[GridArray] = None,
             rtol: float = 1e-6,
             atol: float = 1e-6,
             maxiter: Optional[int] = None) -> GridArray:
  """Conjugate gradient solve for the pressure such that continuity is enforced.

  Returns a pressure correction `q` such that `div(v - grad(q)) == 0`.

  The relationship between `q` and our actual pressure estimate is given by
  `p = q * density / dt`.

  Args:
    v: the velocity field.
    q0: an initial value, or "guess" for the pressure correction. A common
      choice is the correction from the previous time step.
    rtol: relative tolerance for convergence.
    atol: absolute tolerance for convergence.
    maxiter: optional int, the maximum number of iterations to perform.

  Returns:
    A pressure correction `q` such that `div(v - grad(q))` is zero.
  """
  # TODO(jamieas): add functionality for non-uniform density.
  rhs = fd.divergence(v)
  if q0 is None:
    q0 = grids.applied(jnp.zeros_like)(rhs)
  q, _ = jax.scipy.sparse.linalg.cg(
      fd.laplacian, rhs, x0=q0, tol=rtol, atol=atol, maxiter=maxiter)
  return q


def solve_fast_diag(v: Sequence[GridArray],
                    q0: Optional[GridArray] = None,
                    implementation: Optional[str] = None) -> GridArray:
  """Solve for pressure using the fast diagonalization approach."""
  del q0  # unused
  rhs = fd.divergence(v)
  grid = grids.consistent_grid(*v)
  laplacians = list(map(array_utils.laplacian_matrix, grid.shape, grid.step))
  pinv = fast_diagonalization.psuedoinverse(
      laplacians, rhs.dtype,
      hermitian=True, circulant=True, implementation=implementation)
  return grids.applied(pinv)(rhs)


def projection(
    v: GridArrayVector,
    solve: Callable = solve_fast_diag,
) -> GridArrayVector:
  """Apply pressure projection to make a velocity field divergence free."""
  q = solve(v)
  q_grad = fd.forward_difference(q)
  projected = tuple(u - q_g for u, q_g in zip(v, q_grad))
  return projected
