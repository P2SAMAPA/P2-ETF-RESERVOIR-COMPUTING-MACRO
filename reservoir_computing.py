import numpy as np
from sklearn.linear_model import Ridge

def generate_reservoir(size, connectivity=0.1, spectral_radius=0.9, random_seed=42):
    """Generate random reservoir matrix W with given spectral radius."""
    np.random.seed(random_seed)
    W = np.random.randn(size, size) * (connectivity / np.sqrt(size))
    mask = np.random.rand(size, size) < connectivity
    W = W * mask
    eigvals = np.linalg.eigvals(W)
    rho = np.max(np.abs(eigvals))
    if rho > 0:
        W = W * (spectral_radius / rho)
    return W

def reservoir_computing_score(returns, macro_value, reservoir_size=100, base_radius=0.9, radius_range=0.5, connectivity=0.1, ridge_alpha=1e-4):
    """
    Train an Echo State Network with spectral radius adapted to macro.
    Returns the readout weight (score) for the ETF.
    """
    if len(returns) < 20:
        return 0.0
    # Adapt spectral radius based on macro value
    macro_norm = max(0.0, min(1.0, (macro_value - 10) / 40.0))
    spectral_radius = base_radius - radius_range * macro_norm
    spectral_radius = max(0.1, min(0.99, spectral_radius))
    # Generate reservoir
    W = generate_reservoir(reservoir_size, connectivity, spectral_radius)
    Win = np.random.randn(reservoir_size, 1) * 0.1
    # Run reservoir
    T = len(returns)
    states = np.zeros((reservoir_size, T))
    for t in range(1, T):
        states[:, t] = np.tanh(W @ states[:, t-1] + Win.flatten() * returns[t-1])
    # Drop initial transient
    transient = 10
    X = states[:, transient:].T
    y = returns[transient:]
    # Remove rows with NaN (should not happen, but safe)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    if len(X) < 5:
        return 0.0
    # Regularised linear readout
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge.fit(X, y)
    # Predict next return using last state
    last_state = states[:, -1].reshape(1, -1)
    # Ensure last state has no NaN
    if np.any(np.isnan(last_state)):
        return 0.0
    pred_next = ridge.predict(last_state)[0]
    return float(pred_next)
