# lorenz-chaos-ml

Predicting the Lorenz attractor — and finding out exactly where ML breaks down.

## What This Is

The Lorenz system is one of the most studied examples of deterministic chaos. Its defining property is *sensitive dependence on initial conditions*: two trajectories that start arbitrarily close diverge exponentially fast, at a rate governed by the Largest Lyapunov Exponent (LLE).

This project asks a concrete question: if you train ML models on the attractor's trajectory, how far into the future can they predict before chaos wins?

**Key finding:** Computed LLE ≈ 0.88 via Rosenstein (1993), consistent with the theoretical value of ~0.9, giving a prediction horizon of ~1.14 time units. All three models lose predictive skill well before that limit — LSTM holds longest (~0.5 t), linear regression collapses first (~0.2 t). The gap reflects compounding autoregressive error: a practical ceiling that sits strictly below the mathematical one.

---

## Pipeline

```
simulate.py → features.py → models.py → visualize.py
```

| Module | Role |
|---|---|
| `simulate.py` | Integrates Lorenz ODEs via `scipy.integrate.odeint` (σ=10, ρ=28, β=8/3) |
| `features.py` | Computes LLE (Rosenstein 1993), correlation dimension (Grassberger-Procaccia), lag features |
| `models.py` | Trains Linear Regression, Random Forest, and LSTM; runs autoregressive rollout evaluation |
| `visualize.py` | Phase space plots, time series, prediction breakdown curves |

**Modeling details:**
- Lag window: 50 steps; chronological train/test split; `StandardScaler` fit on train only (no leakage)
- LSTM: 2-layer PyTorch, hidden size 64, dropout 0.2, Adam optimizer
- Evaluation: single-step MSE + multi-step autoregressive rollout over random start points

---

## Repository Structure

```
lorenz-chaos-ml/
├── src/
│   ├── simulate.py
│   ├── features.py
│   ├── models.py
│   └── visualize.py
├── notebooks/        # Exploratory Jupyter notebooks (01–04, one per phase)
├── data/             # Generated arrays and CSVs — no external data
└── reports/
    └── figures/      # 150 dpi PNGs exported by visualize.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

Run any module directly for a smoke test:

```bash
python -m src.simulate
python -m src.features
python -m src.models
python -m src.visualize
```

Launch notebooks:

```bash
jupyter notebook notebooks/
```

---

## Stack

`Python 3.11+` · `NumPy` · `SciPy` · `scikit-learn` · `PyTorch` · `SHAP` · `matplotlib` · `Plotly` · `pandas` · `Black`

---

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | Lorenz simulation and visualization | Done |
| 2 | LLE and correlation dimension | Done |
| 3 | ML modeling (LR, RF, LSTM) + rollout evaluation | In progress |
| 4 | SHAP interpretability and final report | Pending |

---

## References

- Lorenz, E.N. (1963). *Deterministic nonperiodic flow.* Journal of Atmospheric Sciences, 20(2), 130–141.
- Rosenstein, M.T., Collins, J.J., De Luca, C.J. (1993). *A practical method for calculating largest Lyapunov exponents from small data sets.* Physica D, 65(1–2), 117–134.
- Grassberger, P., Procaccia, I. (1983). *Characterization of strange attractors.* Physical Review Letters, 50(5), 346.

---

## License

GPL-3.0 — see [LICENSE](LICENSE)
