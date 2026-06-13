# Reservoir Computing with Macro‑Driven Spectral Radius

Implements an Echo State Network (ESN) where the spectral radius of the reservoir matrix is adapted to macro conditions (e.g., higher VIX → smaller radius → more stable dynamics). The per‑ETF score is the linear readout's predicted next‑day return.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Echo State Network with fixed random reservoir
- Spectral radius ρ = base_radius - range * (VIX_normalised)
- Ridge regression readout
- Score = predicted next‑day return
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-reservoir-computing-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast, only linear readout training)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High positive score → ETF expected to rise tomorrow.
- Negative score → expected to fall.

## Requirements

See `requirements.txt`.
