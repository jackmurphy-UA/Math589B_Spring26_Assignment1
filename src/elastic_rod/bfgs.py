from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any

ValueGrad = Callable[[np.ndarray], Tuple[float, np.ndarray]]


@dataclass
class BFGSResult:
    x: np.ndarray
    f: float
    g: np.ndarray
    n_iter: int
    n_feval: int
    converged: bool
    history: Dict[str, Any]


def backtracking_line_search(
    f_and_g: ValueGrad,
    x: np.ndarray,
    f: float,
    g: np.ndarray,
    p: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    tau: float = 0.5,
    max_steps: int = 30,
) -> Tuple[float, float, np.ndarray, int]:
    """
    Simple Armijo backtracking line search.
    Returns (alpha, f_new, g_new, n_feval_increment).

    Armijo condition:
        f(x + a p) <= f(x) + c1 * a * g^T p
    """
    # Ensure we have a descent direction; caller should have ensured this,
    # but we guard anyway.
    gtp = float(np.dot(g, p))
    if not np.isfinite(gtp) or gtp >= 0.0:
        # Fall back to steepest descent direction
        p = -g
        gtp = float(np.dot(g, p))

    alpha = float(alpha0)
    n_feval_inc = 0

    # Precompute RHS slope factor
    # (gtp < 0 for descent)
    for _ in range(max_steps):
        x_new = x + alpha * p
        f_new, g_new = f_and_g(x_new)
        n_feval_inc += 1

        if np.isfinite(f_new) and np.all(np.isfinite(g_new)):
            if f_new <= f + c1 * alpha * gtp:
                return alpha, float(f_new), np.asarray(g_new, dtype=np.float64), n_feval_inc

        alpha *= tau

        # Avoid vanishing steps
        if alpha < 1e-16:
            break

    # If we get here, line search failed; take the (tiny) step we have,
    # but return the last evaluated f,g if finite, else revert to current.
    x_new = x + alpha * p
    f_new, g_new = f_and_g(x_new)
    n_feval_inc += 1
    if not (np.isfinite(f_new) and np.all(np.isfinite(g_new))):
        return 0.0, float(f), np.asarray(g, dtype=np.float64), n_feval_inc
    return alpha, float(f_new), np.asarray(g_new, dtype=np.float64), n_feval_inc


def bfgs(
    f_and_g: ValueGrad,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    alpha0: float = 1.0,
) -> BFGSResult:
    """
    Minimize f(x) with BFGS (inverse-Hessian form).

    Maintains H_k ≈ (∇^2 f(x_k))^{-1}, updates via:
      s_k = x_{k+1} - x_k
      y_k = g_{k+1} - g_k
      ρ_k = 1 / (y_k^T s_k)
      H_{k+1} = (I - ρ s y^T) H_k (I - ρ y s^T) + ρ s s^T

    Includes curvature checks and a descent-direction safeguard.
    """
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()
    f, g = f_and_g(x)
    f = float(f)
    g = np.asarray(g, dtype=np.float64)
    n_feval = 1

    n = x.size
    H = np.eye(n, dtype=np.float64)

    hist: Dict[str, Any] = {
        "f": [f],
        "gnorm": [float(np.linalg.norm(g))],
        "alpha": [],
    }

    for k in range(max_iter):
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            return BFGSResult(
                x=x, f=f, g=g, n_iter=k, n_feval=n_feval,
                converged=True, history=hist
            )

        # Search direction p = -H g
        p = -(H @ g)

        # Safeguard: if not descent, reset H and use steepest descent
        gtp = float(np.dot(g, p))
        if not np.isfinite(gtp) or gtp >= 0.0:
            H[:] = np.eye(n, dtype=np.float64)
            p = -g
            gtp = float(np.dot(g, p))

        # Line search (Armijo)
        alpha, f_new, g_new, inc = backtracking_line_search(
            f_and_g, x, f, g, p, alpha0=alpha0
        )
        n_feval += inc
        hist["alpha"].append(float(alpha))

        # If line search failed completely, stop (non-converged)
        if alpha == 0.0:
            return BFGSResult(
                x=x, f=f, g=g, n_iter=k, n_feval=n_feval,
                converged=False, history=hist
            )

        x_new = x + alpha * p

        s = x_new - x
        y = g_new - g

        # BFGS update with curvature check
        ys = float(np.dot(y, s))
        if np.isfinite(ys) and ys > 1e-12 * float(np.dot(s, s)):
            rho = 1.0 / ys

            # Efficient symmetric rank-2 update:
            # H = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
            Hy = H @ y
            yHy = float(np.dot(y, Hy))

            # Numerically stable form:
            # H_{k+1} = H + (1 + rho*y^T H y) * rho * s s^T - rho*(s Hy^T + Hy s^T)
            H += (1.0 + rho * yHy) * rho * np.outer(s, s) - rho * (np.outer(s, Hy) + np.outer(Hy, s))
        else:
            # If curvature condition fails, reset H (robust, cheap at these sizes)
            H[:] = np.eye(n, dtype=np.float64)

        # Accept step
        x = x_new
        f = float(f_new)
        g = np.asarray(g_new, dtype=np.float64)

        hist["f"].append(f)
        hist["gnorm"].append(float(np.linalg.norm(g)))

    return BFGSResult(
        x=x, f=f, g=g, n_iter=max_iter, n_feval=n_feval,
        converged=False, history=hist
    )
