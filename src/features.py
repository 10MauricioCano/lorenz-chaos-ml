"""
features.py
-----------
Feature engineering for the Lorenz dynamical system.

Computes chaos-theoretic and fractal features from simulated trajectories,
and constructs lag-based supervised learning datasets for ML modeling.

Key features computed:
    - Largest Lyapunov Exponent (LLE): rate of exponential divergence
    - Correlation Dimension: fractal dimension of the strange attractor
    - Lag features: past states as predictors of future states
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Largest Lyapunov Exponent
# ---------------------------------------------------------------------------

def compute_lyapunov_exponent(
    trajectory: np.ndarray,
    dt: float,
    n_steps: int = 1000,
    epsilon: float = 1e-5,
) -> float:
    
    """
    Estimate the largest Lyapunov exponent (LLE) from a single trajectory.
    
    Uses the Rosenstein et al. (1993) nearest-neighbour method: for each point in the trajectory, finds its nearest neighbor,
    track how the distance between them grows over time, then fit a line to the mean log-divergence. The slope is the LLE
    
    A positive LLE confirms chaos. The inverse (1/LLE) approximates the prediction horizon in the same time units as dt
    
    Args:
        trajectory: 2D Array of shape (n_steps, 3) with columns [x, y, z]
        dt: Time step between consecutive states (t_end / n_steps)
        n_steps: Number of steps to track divergence. Default is 1000
        epsilon: Minimun distance threshold to avoid trivially close neighbors 
                (i.e., the same point or adjacent points). Default is 1e-5
                
    Returns:
        Estimated largest Lyapunov exponent as a float.
        Positive value confirms chaotic behavior    
    """
    n = len(trajectory)
    log_divergences = []
    
    for i in range(n - n_steps):
        # Compute distances from point i to all other points
        distances = np.sqrt(np.sum((trajectory - trajectory[i]) ** 2, axis = 1))
        
        # Exclude the point itself and its immediate neighbors (within epsilon)
        distances[:max(1, i - 10)] = np.inf
        distances[i:min(n, i + 10)] = np.inf
        
        # Find nearest neighbor index
        j = np.argmin(distances)
        
        if distances[j] < epsilon or distances[j] == np.inf:
            continue
        
        # Track how the distance evolve over n_steps
        future_i = min(i + n_steps, n)
        future_j = min(j + n_steps, n)
        steps = min(future_i - i, future_j - j)
        
        if steps < 2:
            continue
        
        divergence = np.sqrt(
            np.sum(
                (trajectory[i : i + steps] - trajectory[j : j + steps]) ** 2,
                 axis = 1
            )
        )
        
        #Avoid log(0)
        divergence = np.where(divergence > 0, divergence, np.nan)
        log_div = np.log(divergence)
        
        if not np.all(np.isnan(log_div)):
            log_divergences.append(log_div)
            
            
    if not log_divergences:
        raise ValueError ("Could not compute Largest Lyapunov Exponent: no valid neighbors pairs found")
    
    # Align and average log-divergence curves
    min_len = min(len(d) for d in log_divergences)
    mean_log_div = np.nanmean(
        [d[: min_len] for d in log_divergences], axis = 0
    )
    
    #Fit a line. The slope is the Largest Lyapunov Exponent
    time_axis = np.arange(min_len) * dt
    valid = ~np.isnan(mean_log_div)
    
    reg = LinearRegression()
    reg.fit(time_axis[valid].reshape(-1, 1), mean_log_div[valid])
    
    return float(reg.coef_[0])

# ---------------------------------------------------------------------------
# Correlation Dimension (Fractal Dimension)
# ---------------------------------------------------------------------------

def compute_correlation_dimension(
    trajectory: np.ndarray,
    n_samples: int = 2000,
    r_values: int = 20,
    
) ->tuple[float, np.ndarray, np.ndarray]: 
    """
    
    Estimate  the correlation dimension  of the attractor (Grassberger-Procaccia)
    
    The correlation dimension is a measure of the fractal complexity of the attractor. For the Lorenz System
    the theoretical value is approximately 2.05, meaning it fills slightly more than a 2D surface in 3D space.
    
    the method counts how many pairs of points are within distance r for increasing values of r, then fits
    log(C(r)) ~ D * log(r) to find D
    
    Args:
    
        trajectory: 2D array of shape (n_steps, 3) with columns [x, y, z].
        n_samples: Number of points to subsample for efficiency. Default 2000
        r_values: Number of radius values to evaluate. Default is 20
        
    Returns:
        dimension: Estimated correlation dimension
        log_r: Log of radius values used (for plotting)
        log_c: Log of correlation integral values (for plotting)
    
    """
 # Subsample for computational efficiency
 
    idx = np.random.choice(len(trajectory), size = min(n_samples, len(trajectory))
                            , replace = False) 
    
    pts = trajectory[idx] 

    # Compute all pairwise  distances

    n = len(pts)
    distances = []
    for i in range(n):
        dists = np.sqrt(np.sum((pts[i + 1: ] - pts[i]) ** 2, axis = 1))
        distances.extend(dists)
    distances = np.array(distances)

    # Define radius range from 1st to 99th percentile of distances
    r_min = np.percentile(distances, 1)
    r_max = np.percentile(distances, 99)
    r_range = np.logspace(np.log10(r_min), np.log10(r_max), r_values)

    # Compute correlation integral C(r)
    n_pairs = len(distances)
    c_r = np.array([np.sum(distances < r) / n_pairs for r in r_range])

    # Avoid log(0)
    valid = c_r > 0
    log_r = np.log(r_range[valid])
    log_c = np.log(c_r[valid])

    # Fit only the middle scaling region (trim bottom and top 20%)
    # The edges of the log-log curve are dominated by finite-size effects
    # and saturation — the true fractal dimension lives in the linear middle
    n_valid = len(log_r)
    trim = max(1, int(n_valid * 0.20))
    log_r_mid = log_r[trim:-trim]
    log_c_mid = log_c[trim:-trim]

    reg = LinearRegression()
    reg.fit(log_r_mid.reshape(-1, 1), log_c_mid)
    dimension = float(reg.coef_[0])

    return dimension, log_r, log_c

# ---------------------------------------------------------------------------
# Lag Feature Construction
# ---------------------------------------------------------------------------

def build_lag_features(
    trajectory: np.ndarray,
    n_lags: int = 10,
    target_steps_ahead: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Convert a multivariate time series into a supervised learning dataset.
    
    For each time step t, the input X contains the system state at t-1, t-2, ..., t-n_lags.
    The target y contains the state at t + target_steps_ahead
    
    This framing allows classical ML models (Random Forest, XGboost) to learn the mapping from past
    states to future states.
    
    Args:
        trajectory: 2D array of shape (n_steps, 3) with columns [x, y, z].
        n_lags: Number of past time steps to use as features. Default is 10
        targets_steps_ahead: How many steps into the future to predict. Default is 1 (Next step).
        
    Returns:
        X: feature matrix of shape (n_samples, n_lags * 3).
            Each row contains [x(t-1), y(t-1), z(t-1), ..., x(t-n_lags), ...]
        y: Target matrix of shape (n_samples, 3)
            Each row contains [x(t + ahead), y(t + ahead), z(t + ahead)].
    """
    
    n = len(trajectory)
    X_rows = []
    y_rows = []
    
    for i in range(n_lags, n - target_steps_ahead):
        # Flatten past n_lags states into a single feature vector
        past = trajectory[i - n_lags:i].flatten()
        future = trajectory[i + target_steps_ahead]
        
        X_rows.append(past)
        y_rows.append(future)
        
    X = np.array(X_rows)
    y = np.array(y_rows)
    
    return X, y

# ---------------------------------------------------------------------------
# Entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    from simulate import simulate_lorenz
    
    print ("Simulating Lorenz System")
    t, trajectory = simulate_lorenz(t_end= 100, n_steps= 50000)
    dt = t[1] -t[0]
    
    print("\nComputing Largest Lyapunov Exponent")
    lle = compute_lyapunov_exponent(trajectory, dt = dt, n_steps= 500)
    print(f'LLE ≈ {lle:.4f} (Theoretical ≈ 0.9)')
    print(f'Prediction horizon ≈ {1/lle:.4f} time units')
    
    print("\nComputing Correlation Dimension")
    dim, log_r, log_c = compute_correlation_dimension(trajectory, n_samples= 3000)
    print(f'Correlation dimension ≈ {dim:.4f} (Theoretical ≈ 2.05)')
    
    print("\nBuilding lag features")
    X, y = build_lag_features(trajectory, n_lags= 10, target_steps_ahead= 1)
    print(f'X Shape: {X.shape}')
    print(f'y Shape: {y.shape}')
    
    print("\nFeature engineering complete")
    
    