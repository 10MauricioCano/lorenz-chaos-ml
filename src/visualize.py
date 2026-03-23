"""
visualize.py
------------
Visualization functions for the Lorenz dynamical system.

Provides 2D and 3D plotting utilities for Lorenz trajectories.
All functions are designed to be imported by notebooks and scripts.
Figures can be displayed interactively or saved to disk.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — required for 3D projection


# ---------------------------------------------------------------------------
# 3D Attractor Plot
# ---------------------------------------------------------------------------

def plot_attractor_3d(
    trajectory: np.ndarray,
    title: str = "Lorenz Attractor",
    color_by_time: bool = True,
    save_path: str | None = None,
) -> None:
    """
    Plot the full Lorenz trajectory in 3D phase space.

    The trajectory is colored by time progression when color_by_time=True,
    making it easy to see how the system evolves from start to finish.

    Args:
        trajectory: 2D array of shape (n_steps, 3) with columns [x, y, z].
        title: Title for the plot. Default is 'Lorenz Attractor'.
        color_by_time: If True, color the trajectory by time (early=blue,
                       late=red). If False, use a single color. Default True.
        save_path: If provided, saves the figure to this path instead of
                   displaying it. Example: 'reports/figures/attractor_3d.png'

    Returns:
        None
    """
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    if color_by_time:
        # Segment the trajectory into chunks and color each progressively
        n_segments = 500
        segment_size = len(x) // n_segments
        colormap = plt.cm.plasma

        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size + 1
            color = colormap(i / n_segments)
            ax.plot(x[start:end], y[start:end], z[start:end],
                    color=color, linewidth=0.4, alpha=0.85)
    else:
        ax.plot(x, y, z, linewidth=0.4, alpha=0.85, color="steelblue")

    # Mark the initial condition
    ax.scatter(x[0], y[0], z[0], color="green", s=30, zorder=5, label="Start")
    ax.scatter(x[-1], y[-1], z[-1], color="red", s=30, zorder=5, label="End")

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------------
# 2D Projections
# ---------------------------------------------------------------------------

def plot_attractor_2d(
    trajectory: np.ndarray,
    save_path: str | None = None,
) -> None:
    """
    Plot the three 2D projections of the Lorenz attractor (XY, XZ, YZ).

    Projections are useful for understanding the geometric structure of
    the attractor from different perspectives without 3D rendering.

    Args:
        trajectory: 2D array of shape (n_steps, 3) with columns [x, y, z].
        save_path: If provided, saves the figure to this path.

    Returns:
        None
    """
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Lorenz Attractor — 2D Projections", fontsize=14)

    projections = [
        (x, y, "X", "Y", "XY Plane"),
        (x, z, "X", "Z", "XZ Plane"),
        (y, z, "Y", "Z", "YZ Plane"),
    ]

    for ax, (h, v, xlabel, ylabel, subtitle) in zip(axes, projections):
        ax.plot(h, v, linewidth=0.3, alpha=0.7, color="steelblue")
        ax.set_title(subtitle, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------------
# Time Series Plot
# ---------------------------------------------------------------------------

def plot_time_series(
    t: np.ndarray,
    trajectory: np.ndarray,
    t_max: float = 30.0,
    save_path: str | None = None,
) -> None:
    """
    Plot x, y, z coordinates as time series up to t_max.

    This view reveals the oscillatory and irregular temporal behavior
    of the Lorenz system — the signature of chaos in the time domain.

    Args:
        t: 1D array of time values, shape (n_steps,).
        trajectory: 2D array of shape (n_steps, 3) with columns [x, y, z].
        t_max: Maximum time to display. Useful to zoom into early dynamics.
               Default is 30.0.
        save_path: If provided, saves the figure to this path.

    Returns:
        None
    """
    mask = t <= t_max
    t_plot = t[mask]
    traj_plot = trajectory[mask]

    labels = ["X(t)", "Y(t)", "Z(t)"]
    colors = ["steelblue", "tomato", "seagreen"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Lorenz System — Time Series", fontsize=14)

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(t_plot, traj_plot[:, i], linewidth=0.8, color=color)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------------
# Entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from simulate import simulate_lorenz

    print("Running simulation...")
    t, trajectory = simulate_lorenz()

    print("Rendering 3D attractor...")
    plot_attractor_3d(trajectory, save_path="reports/figures/attractor_3d.png")

    print("Rendering 2D projections...")
    plot_attractor_2d(trajectory, save_path="reports/figures/attractor_2d.png")

    print("Rendering time series...")
    plot_time_series(t, trajectory, save_path="reports/figures/time_series.png")

    print("All figures saved to reports/figures/")