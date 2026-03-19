"""
simulate.py
-----------
Numerical simulation of the Lorenz dynamical system.

The Lorenz system is a set of three coupled ordinary differential equations
originally derived by Edward Lorenz (1963) as a simplified model of
atmospheric convection. It is one of the canonical examples of deterministic
chaos: small differences in initial conditions lead to exponentially diverging
trajectories.

Equations:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

Classic chaotic parameters:
    sigma = 10.0
    rho   = 28.0
    beta  = 8/3
"""

import numpy as np
from scipy.integrate import odeint


# ---------------------------------------------------------------------------
# Core system definition
# ---------------------------------------------------------------------------

def lorenz_system(
    state: list[float],
    t: float,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8 / 3,
) -> list[float]:
    """
    Compute the time derivatives of the Lorenz system at a given state.

    This function defines the right-hand side of the Lorenz ODEs and is
    designed to be passed directly to scipy.integrate.odeint.

    Args:
        state: Current system state as [x, y, z].
        t: Current time (required by odeint signature, not used explicitly).
        sigma: Prandtl number. Controls the ratio of fluid viscosity to
               thermal diffusivity. Default is 10.0.
        rho: Rayleigh number. Controls the temperature difference driving
             convection. Default is 28.0 (chaotic regime).
        beta: Geometric factor. Default is 8/3.

    Returns:
        List of derivatives [dx/dt, dy/dt, dz/dt] at the given state.
    """
    x, y, z = state

    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    return [dx_dt, dy_dt, dz_dt]


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def simulate_lorenz(
    initial_state: list[float] = [0.0, 1.0, 0.0],
    t_start: float = 0.0,
    t_end: float = 50.0,
    n_steps: int = 10000,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8 / 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the Lorenz system over a time interval.

    Numerically integrates the Lorenz ordinary diferential equations  using scipy's odeint solver,
    which implements the LSODA algorithm — an adaptive step-size method
    that automatically switches between stiff and non-stiff solvers.

    Args:
        initial_state: Starting point [x0, y0, z0] in phase space.
                       Default is [0.0, 1.0, 0.0].
        t_start: Start time of the simulation. Default is 0.0.
        t_end: End time of the simulation. Default is 50.0.
        n_steps: Number of time points to evaluate. Higher values give
                 smoother trajectories. Default is 10000.
        sigma: Lorenz parameter sigma. Default is 10.0.
        rho: Lorenz parameter rho. Default is 28.0.
        beta: Lorenz parameter beta. Default is 8/3.

    Returns:
        t: 1D numpy array of shape (n_steps,) with time values.
        trajectory: 2D numpy array of shape (n_steps, 3) where each row
                    is [x, y, z] at the corresponding time step.
    """
    t = np.linspace(t_start, t_end, n_steps)

    trajectory = odeint(
        func=lorenz_system,
        y0=initial_state,
        t=t,
        args=(sigma, rho, beta),
    )

    return t, trajectory


# ---------------------------------------------------------------------------
# Entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t, trajectory = simulate_lorenz()

    print(f"Simulation complete.")
    print(f"  Time steps : {len(t)}")
    print(f"  Duration   : {t[-1]:.1f} time units")
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  First state : x={trajectory[0,0]:.4f}, y={trajectory[0,1]:.4f}, z={trajectory[0,2]:.4f}")
    print(f"  Final state : x={trajectory[-1,0]:.4f}, y={trajectory[-1,1]:.4f}, z={trajectory[-1,2]:.4f}")