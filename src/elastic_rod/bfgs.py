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
    tau: float = 0.6,
    max_steps: int = 20,
) -> Tuple[float, float, np.ndarray, int]:
    """
    Armijo backtracking line search with a safe fallback:
    - Try to satisfy Armijo.
    - If not found within budget, return the best (lowest f) tried step
      as long as it improved f. Otherwise, take a tiny descent step.
    """
    gtp = float(np.dot(g, p))
    if not np.isfinite(gtp) or gtp >= 0.0:
        p = -g
        gtp = float(np.dot(g, p))
        if not np.isfinite(gtp) or gtp >= 0.0:
            # can't find a descent direction
            return 0.0, float(f), np.asarray(g, dtype=np.float64), 0

    alpha = float(alpha0)
    n_feval_inc = 0

    best_alpha = 0.0
    best_f = float(f)
    best_g = np.asarray(g, dtype=np.float64)

    for _ in range(max_steps):
        x_new = x + alpha * p
        f_new, g_new = f_and_g(x_new)
        n_feval_inc += 1

        if np.isfinite(f_new) and np.all(np.isfinite(g_new)):
            f_new = float(f_new)
            g_new = np.asarray(g_new, dtype=np.float64)

            # Track best seen (even if Armijo fails)
            if f_new < best_f:
                best_f = f_new
                best_alpha = alpha
                best_g = g_new

            # Armijo accept
            if f_new <= f + c1 * alpha * gtp:
                return alpha, f_new, g_new, n_feval_inc

        alpha *= tau
        if alpha < 1e-16:
            break

    # Fallback: take best improvement if we found one
    if best_alpha > 0.0 and best_f < f:
        return best_alpha, best_f, best_g, n_feval_inc

    # Last-resort: tiny step along descent direction (guarantees some movement)
    alpha = 1e-6
    x_new = x + alpha * p
    f_new, g_new = f_and_g(x_new)
    n_feval_inc += 1
    if np.isfinite(f_new) and np.all(np.isfinite(g_new)):
        return alpha, float(f_new), np.asarray(g_new, dtype=np.float64), n_feval_inc

    return 0.0, float(f), np.asarray(g, dtype=np.float64), n_feval_inc

def bfgs(
    f_and_g: ValueGrad,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    alpha0: float = 1.0,
) -> BFGSResult:
    """
    Inverse-Hessian BFGS with Armijo line search and curvature safeguards.
    """
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()
    f, g = f_and_g(x)
    f = float(f)
    g = np.asarray(g, dtype=np.float64)
    n_feval = 1

    n = x.size
    H = np.eye(n, dtype=np.float64)

    hist: Dict[str, Any] = {"f": [f], "gnorm": [float(np.linalg.norm(g))], "alpha": []}

    alpha = float(alpha0)  # warm-start line search

    for k in range(max_iter):
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            return BFGSResult(x=x, f=f, g=g, n_iter=k, n_feval=n_feval, converged=True, history=hist)

        p = -(H @ g)

        # ensure descent
        gtp = float(np.dot(g, p))
        if not np.isfinite(gtp) or gtp >= 0.0:
            H[:] = np.eye(n, dtype=np.float64)
            p = -g

        alpha, f_new, g_new, inc = backtracking_line_search(
            f_and_g, x, f, g, p, alpha0=alpha
        )
        n_feval += inc
        hist["alpha"].append(float(alpha))

        if alpha == 0.0:
            return BFGSResult(x=x, f=f, g=g, n_iter=k, n_feval=n_feval, converged=False, history=hist)

        x_new = x + alpha * p
        s = x_new - x
        y = g_new - g

        ys = float(np.dot(y, s))
        sTs = float(np.dot(s, s))

        if np.isfinite(ys) and ys > 1e-14 * sTs:
            rho = 1.0 / ys
            Hy = H @ y
            yHy = float(np.dot(y, Hy))
            H += (1.0 + rho * yHy) * rho * np.outer(s, s) - rho * (np.outer(s, Hy) + np.outer(Hy, s))
        else:
            if (not np.isfinite(ys)) or (ys <= 0.0):
                H[:] = np.eye(n, dtype=np.float64)
            # else: keep H unchanged

        x = x_new
        f = float(f_new)
        g = np.asarray(g_new, dtype=np.float64)

        hist["f"].append(f)
        hist["gnorm"].append(float(np.linalg.norm(g)))

        # warm start next iteration line search
        alpha = min(1.0, 1.1 * alpha)

    return BFGSResult(x=x, f=f, g=g, n_iter=max_iter, n_feval=n_feval, converged=False, history=hist)
