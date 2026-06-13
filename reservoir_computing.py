import numpy as np
from sklearn.linear_model import Ridge

def generate_reservoir(size, connectivity=0.1, spectral_radius=0.9, random_seed=42):
    """Generate random reservoir matrix W with given spectral radius."""
    np.random.seed(random_seed)
    W = np.random.randn(size, size) * (connectivity / np.sqrt(size))
    # Set sparse entries to zero based on connectivity
    mask = np.random.rand(size, size) < connectivity
    W = W * mask
    # Scale to desired spectral radius
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
    # Adapt spectral radius based on macro value (e.g., VIX)
    # Normalise macro: typical VIX range 10-40
    macro_norm = max(0.0, min(1.0, (macro_value - 10) / 40.0))
    # Higher macro -> smaller spectral radius (more stable)
    spectral_radius = base_radius - radius_range * macro_norm
    spectral_radius = max(0.1, min(0.99, spectral_radius))
    # Generate reservoir
    W = generate_reservoir(reservoir_size, connectivity, spectral_radius)
    # Input weights (random)
    Win = np.random.randn(reservoir_size, 1) * 0.1
    # Run reservoir
    T = len(returns)
    states = np.zeros((reservoir_size, T))
    for t in range(1, T):
        states[:, t] = np.tanh(W @ states[:, t-1] + Win.flatten() * returns[t-1])
    # Collect states (drop initial transient)
    transient = 10
    X = states[:, transient:].T
    y = returns[transient:]
    # Regularised linear readout
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge.fit(X, y)
    # Score: the weight corresponding to the last state? Actually we want the predicted return for tomorrow.
    # We'll use the last state to predict next return.
    last_state = states[:, -1].reshape(1, -1)
    pred_next = ridge.predict(last_state)[0]
    return float(pred_next)
