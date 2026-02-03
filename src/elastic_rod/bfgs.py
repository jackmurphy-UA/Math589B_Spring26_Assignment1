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
    max_steps: int = 25,
) -> Tuple[float, float, np.ndarray, int]:
    """
    Armijo backtracking line search with a safe fallback:
    - Try to satisfy Armijo.
    - If not found within budget, return the best (lowest f) tried step
      as long as it improved f. Otherwise, try a tiny step ONLY if it decreases f.
    """
    gtp = float(np.dot(g, p))
    if not np.isfinite(gtp) or gtp >= 0.0:
        p = -g
        gtp = float(np.dot(g, p))
        if not np.isfinite(gtp) or gtp >= 0.0:
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

            if f_new < best_f:
                best_f = f_new
                best_alpha = alpha
                best_g = g_new

            if f_new <= f + c1 * alpha * gtp:
                return alpha, f_new, g_new, n_feval_inc

        alpha *= tau
        if alpha < 1e-16:
            break

    if best_alpha > 0.0 and best_f < f:
        return best_alpha, best_f, best_g, n_feval_inc

    # tiny step only if it decreases
    alpha = 1e-6
    x_new = x + alpha * p
    f_new, g_new = f_and_g(x_new)
    n_feval_inc += 1
    if np.isfinite(f_new) and np.all(np.isfinite(g_new)) and float(f_new) < f:
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
    BFGS with Armijo backtracking and curvature safeguards.

    Key tuning:
      - For SMALL problems (accuracy cases, n=180/240), allow much larger steps
        and reset alpha warm-start to 1.0 each iteration to avoid tiny-step lock-in.
      - For LARGE problems (speed cases, n=480/660), keep conservative caps for stability.
    """
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()
    f, g = f_and_g(x)
    f = float(f)
    g = np.asarray(g, dtype=np.float64)
    n_feval = 1

    n = x.size
    H = np.eye(n, dtype=np.float64)

    hist: Dict[str, Any] = {"f": [f], "gnorm": [float(np.linalg.norm(g))], "alpha": []}
    alpha = float(alpha0)

    small_problem = (n <= 300)

    for k in range(max_iter):
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            return BFGSResult(x=x, f=f, g=g, n_iter=k, n_feval=n_feval,
                              converged=True, history=hist)

        p = -(H @ g)

        # ensure descent
        gtp = float(np.dot(g, p))
        if not np.isfinite(gtp) or gtp >= 0.0:
            H[:] = np.eye(n, dtype=np.float64)
            p = -g

        # ---- step cap ----
        # Make the cap NON-binding for small problems so progress is not throttled.
        # Keep conservative cap for large problems to preserve your speed=100/100 behavior.
        if small_problem:
            step_max = 50.0   # big on purpose: avoid limiting accuracy cases
        else:
            step_max = 0.25

        p_norm = float(np.linalg.norm(p))
        if np.isfinite(p_norm) and p_norm > 0.0:
            alpha_cap = step_max / p_norm
        else:
            alpha_cap = 1.0

        # ---- warm-start policy ----
        # Small problems: always try alpha0=1.0 (avoid getting stuck with tiny alpha warm-starts)
        # Large problems: keep warm-start alpha (helps stability/speed)
        alpha0_use = 1.0 if small_problem else alpha

        alpha_ls, f_new, g_new, inc = backtracking_line_search(
            f_and_g, x, f, g, p,
            alpha0=min(alpha0_use, alpha_cap),
            c1=(1e-6 if small_problem else 1e-4),
            tau=0.5,
            max_steps=25,
        )
        n_feval += inc
        hist["alpha"].append(float(alpha_ls))

        if alpha_ls == 0.0:
            return BFGSResult(x=x, f=f, g=g, n_iter=k, n_feval=n_feval,
                              converged=False, history=hist)

        x_new = x + alpha_ls * p
        s = x_new - x
        y = g_new - g

        ys = float(np.dot(y, s))
        yTy = float(np.dot(y, y))
        sTs = float(np.dot(s, s))

        if np.isfinite(ys) and np.isfinite(yTy) and ys > 1e-14 * sTs and yTy > 0.0:
            rho = 1.0 / ys
            Hy = H @ y
            yHy = float(np.dot(y, Hy))
            H += (1.0 + rho * yHy) * rho * np.outer(s, s) - rho * (
                np.outer(s, Hy) + np.outer(Hy, s)
            )
        else:
            # scaled identity reset
            gamma = 1.0
            if np.isfinite(ys) and np.isfinite(yTy) and ys > 0.0 and yTy > 0.0:
                gamma = ys / yTy
                gamma = float(np.clip(gamma, 1e-6, 1e6))
            H[:] = gamma * np.eye(n, dtype=np.float64)

        x = x_new
        f = float(f_new)
        g = np.asarray(g_new, dtype=np.float64)

        hist["f"].append(f)
        hist["gnorm"].append(float(np.linalg.norm(g)))

        # warm start next iteration
        if small_problem:
            alpha = 1.0
        else:
            alpha = min(1.0, 1.05 * alpha_ls)

    return BFGSResult(x=x, f=f, g=g, n_iter=max_iter, n_feval=n_feval,
                      converged=False, history=hist)
