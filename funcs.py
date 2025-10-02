import numpy as np
from dataclasses import dataclass

@dataclass
class Solution:
    p_grid: np.ndarray          # shape (M,)
    t_grid: np.ndarray          # shape (N+1,)
    V: np.ndarray               # shape (N+1, M)   V[n,i] = V(p_i, t_n)
    g: np.ndarray               # shape (N+1, M)
    C_mask: np.ndarray          # shape (N+1, M), True if in continuation
    p_left: np.ndarray          # shape (N+1,), left boundary (nan if none)
    p_right: np.ndarray         # shape (N+1,), right boundary (nan if none)

def solve_continuation_region(
    mu, sigma, f0, f1, p0,
    T=1.0, mu0=0.0,
    M=401, N=400,
    omega=1.5, tol=1e-10, max_iter=20000
) -> Solution:
    """
    Variational-inequality solver for V_t + a(p) V_pp = 0 with obstacle g(p,t).
    Discretization: time-to-go s = T - t (implicit Euler), centered space.
    LCP solved by PSOR at each step.
    """
    # grids
    p = np.linspace(0.0, 1.0, M)
    dp = p[1] - p[0]
    s_grid = np.linspace(0.0, T, N+1)        # s = 0..T
    t_grid = T - s_grid                      # t descends T..0
    ds = s_grid[1] - s_grid[0]

    # coefficients a(p)
    a = 2.0 * (mu / sigma) ** 2 * (p * (1.0 - p)) ** 2
    lam = ds * a / (dp ** 2)                 # local CFL-like number

    # obstacle g(p,t) on the whole grid
    def g_of(pv, t):
        risky = (2.0 * pv - 1.0) * mu + f1(t)
        safe  = mu0 + f0(T - t)
        return np.maximum(risky, safe)

    g = np.zeros((N+1, M))
    for n, t in enumerate(t_grid):
        g[n, :] = g_of(p, t)

    # allocate value array; initial condition at s=0 (t=T)
    V = np.zeros_like(g)
    V[0, :] = g[0, :]

    # PSOR solver for the LCP: (M x >= q, x >= obstacle, complementarity)
    # Here M = I - ds*A, q = V_prev
    def psor_step(V_prev, g_now):
        x = V_prev.copy()   # initial guess
        diag = 1.0 + 2.0 * lam
        lower = -lam[1:]    # for i >=1, coefficient on i-1
        upper = -lam[:-1]   # for i <= M-2, coefficient on i+1

        for it in range(max_iter):
            x_old_inf = x.copy()  # for infinity-norm check (cheap)

            # i = 0 (boundary): lam[0]=0, so M_00 = 1, b = V_prev[0]
            i = 0
            Mi = diag[i]
            # sum = Mi * x[i] + upper[i]*x[i+1], but upper[0] = -lam[0] = 0
            sum_i = Mi * x[i]
            y = x[i] + omega * (V_prev[i] - sum_i) / Mi
            x[i] = max(g_now[i], y)

            # interior nodes
            for i in range(1, M-1):
                Mi = diag[i]
                sum_i = lower[i-1] * x[i-1] + Mi * x[i] + upper[i] * x[i+1]
                y = x[i] + omega * (V_prev[i] - sum_i) / Mi
                # projection on obstacle
                x[i] = max(g_now[i], y)

            # i = M-1 (boundary): lam[-1]=0, so M_nn = 1
            i = M-1
            Mi = diag[i]
            sum_i = Mi * x[i]
            y = x[i] + omega * (V_prev[i] - sum_i) / Mi
            x[i] = max(g_now[i], y)

            # convergence check (∞-norm)
            err = np.max(np.abs(x - x_old_inf))
            if err < tol:
                break
        else:
            raise RuntimeError("PSOR did not converge; try increasing max_iter or adjusting omega.")

        return x

    # march in s (increasing), i.e. backward in t
    for n in range(1, N+1):
        V[n, :] = psor_step(V[n-1, :], g[n, :])

    # continuation mask and free boundaries
    C_mask = V > g + 1e-12

    p_left = np.full(N+1, np.nan)
    p_right = np.full(N+1, np.nan)
    
    for n in range(N+1):
        diff = V[n, :] - g[n, :]
        pos = np.where(diff > 0.0)[0]
        if pos.size == 0:
            continue
        iL, iR = pos[0], pos[-1]
        # left boundary interpolation (if not at left edge)
        if iL > 0:
            dl, dr = diff[iL-1], diff[iL]
            p_left[n] = p[iL-1] + (0.0 - dl) * (p[iL] - p[iL-1]) / (dr - dl)
        else:
            p_left[n] = p[iL]
        # right boundary interpolation
        if iR < M-1:
            dl, dr = diff[iR], diff[iR+1]
            p_right[n] = p[iR] + (0.0 - dl) * (p[iR+1] - p[iR]) / (dr - dl)
        else:
            p_right[n] = p[iR]

    return Solution(p_grid=p, t_grid=t_grid, V=V, g=g, C_mask=C_mask, p_left=p_left, p_right=p_right)

def validate_solution_basic(sol, mu, sigma, f0, f1, mu0=0.0, T=None,
                            tol_obs=1e-9, tol_lcp=1e-7, tol_time=1e-9):
    """
    Conservative pass/fail checks (avoid false negatives).

    Returns dict of booleans:
      - no_nans
      - obstacle_feasible
      - terminal_match
      - monotone_in_time_to_go
      - lcp_feasible
      - all_tests_pass
    """
    p = sol.p_grid
    t = sol.t_grid
    V = sol.V
    g = sol.g

    # 0) finite
    no_nans = all(np.isfinite(arr).all() for arr in [p, t, V, g])

    # 1) obstacle feasibility
    obstacle_feasible = (np.min(V - g) >= -tol_obs)

    # 2) terminal match (first slice corresponds to t = T in the provided solver)
    terminal_match = (np.max(np.abs(V[0, :] - g[0, :])) <= tol_obs)

    # 3) monotonicity in time-to-go s (values non-decreasing along marching index)
    if V.shape[0] >= 2:
        dV = V[1:, :] - V[:-1, :]
        monotone_in_time_to_go = (np.min(dV) >= -tol_time)
    else:
        monotone_in_time_to_go = True

    # 4) LCP feasibility: (I - ds*A) V^n - V^{n-1} >= -tol_lcp at all nodes
    if V.shape[0] >= 2 and p.size >= 2:
        ds = float(abs(t[1] - t[0]))               # uniform time step in s
        dp = float(p[1] - p[0])                    # uniform space step
        a = 2.0 * (mu / sigma) ** 2 * (p * (1.0 - p)) ** 2
        lam = ds * a / (dp ** 2)                   # local λ_i

        lcp_feasible = True
        for n in range(1, V.shape[0]):
            Vn = V[n, :]
            Vprev = V[n-1, :]

            # compute r = (I - ds*A)V^n - V^{n-1} using the same stencil as solver
            r = np.empty_like(Vn)
            # left boundary i=0
            r[0] = (1.0 + 2.0 * lam[0]) * Vn[0] - lam[0] * Vn[1] - Vprev[0]
            # interior 1..M-2
            if p.size > 2:
                lam_mid = lam[1:-1]
                r[1:-1] = (-lam_mid * Vn[:-2]
                           + (1.0 + 2.0 * lam_mid) * Vn[1:-1]
                           - lam_mid * Vn[2:]) - Vprev[1:-1]
            # right boundary i=M-1
            r[-1] = -lam[-1] * Vn[-2] + (1.0 + 2.0 * lam[-1]) * Vn[-1] - Vprev[-1]

            if np.min(r) < -tol_lcp:
                lcp_feasible = False
                break
    else:
        lcp_feasible = True

    out = {
        "no_nans": no_nans,
        "obstacle_feasible": obstacle_feasible,
        "terminal_match": terminal_match,
        "monotone_in_time_to_go": monotone_in_time_to_go,
        "lcp_feasible": lcp_feasible,
    }
    out["all_tests_pass"] = all(out.values())
    return out
