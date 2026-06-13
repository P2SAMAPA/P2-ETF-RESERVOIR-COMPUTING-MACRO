import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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

def compute_composite_macro_factor(macro_df, target_returns=None):
    """
    Compute a composite macro factor as a weighted sum of all macro variables.
    If target_returns is provided, weights are estimated via ridge regression.
    Otherwise, equal weights.
    """
    # Remove NaN rows
    if target_returns is not None:
        mask = ~(np.isnan(target_returns) | np.isnan(macro_df).any(axis=1))
        macro_clean = macro_df[mask]
        target_clean = target_returns[mask]
    else:
        macro_clean = macro_df
        target_clean = None
    if target_clean is not None and len(target_clean) > 5:
        # Standardise macro
        scaler = StandardScaler()
        macro_scaled = scaler.fit_transform(macro_clean)
        # Ridge regression to estimate importance
        ridge = Ridge(alpha=1.0)
        ridge.fit(macro_scaled, target_clean)
        weights = np.abs(ridge.coef_)
        weights = weights / (weights.sum() + 1e-8)
    else:
        weights = np.ones(macro_df.shape[1]) / macro_df.shape[1]
        scaler = StandardScaler()
        macro_scaled = scaler.fit_transform(macro_df)
    return weights, scaler

def composite_macro_factor_at_time(macro_row, weights, scaler):
    """Compute composite macro factor for a single row of macro data."""
    macro_scaled = scaler.transform(macro_row.reshape(1, -1)).flatten()
    factor = np.dot(weights, macro_scaled)
    # Normalise to [0,1] range using logistic
    return 1.0 / (1.0 + np.exp(-factor))

def reservoir_computing_score(returns, macro_df, reservoir_size=100, base_radius=0.9, radius_range=0.5, connectivity=0.1, ridge_alpha=1e-4):
    """
    Train an Echo State Network with spectral radius adapted to a composite macro factor.
    Returns the predicted next‑day return.
    """
    if len(returns) < 20 or macro_df is None or len(macro_df) < 20:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Compute macro weights using ridge regression of returns on macro (lagged)
    target = returns[1:]
    macro_lagged = macro_df.iloc[:-1]
    if len(target) != len(macro_lagged):
        min_len2 = min(len(target), len(macro_lagged))
        target = target[:min_len2]
        macro_lagged = macro_lagged.iloc[:min_len2]
    if len(target) > 5:
        weights, scaler = compute_composite_macro_factor(macro_lagged, target)
    else:
        weights = np.ones(macro_df.shape[1]) / macro_df.shape[1]
        scaler = StandardScaler()
        scaler.fit(macro_df)
    # Current macro (last row) to compute factor
    current_macro = macro_df.iloc[-1].values
    macro_factor = composite_macro_factor_at_time(current_macro, weights, scaler)
    # Adapt spectral radius based on macro factor
    spectral_radius = base_radius - radius_range * macro_factor
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
    # Remove NaN rows
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    if len(X) < 5:
        return 0.0
    # Ridge readout
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge.fit(X, y)
    # Predict next return using last state
    last_state = states[:, -1].reshape(1, -1)
    if np.any(np.isnan(last_state)):
        return 0.0
    pred_next = ridge.predict(last_state)[0]
    return float(pred_next)
