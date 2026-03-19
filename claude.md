# CLAUDE.md — Project Instructions for AI Pair Programming

## Project Summary
This project explores the intersection of chaos theory and machine learning.
We simulate the Lorenz attractor system, engineer fractal and chaos-theoretic
features (fractal dimension, Lyapunov exponent), and train ML models to predict
future states — then analyze where and why prediction breaks down relative to
the theoretical chaos horizon.

This is Portfolio Project #1 by Mauricio Cano.

## Tech Stack
- Python 3.11+
- numpy, scipy — simulation and numerical computing
- matplotlib, plotly — visualization
- pandas — data structuring
- scikit-learn — classical ML models
- tensorflow or pytorch — LSTM (decided in Phase 3)
- shap — model interpretability
- jupyter — exploratory notebooks

## Folder Structure
- src/          → Clean, modular Python scripts (no notebooks here)
- notebooks/    → Exploratory Jupyter notebooks, numbered sequentially
- data/         → Only generated data (numpy arrays, CSVs). No raw external data.
- reports/      → Exported figures and findings

## Coding Conventions
- All functions must have docstrings (Google style)
- Use type hints on all function signatures
- Variable names in English, comments in English
- Max line length: 88 characters (Black formatter standard)
- Every script must be importable (use if __name__ == "__main__": guards)

## My Learning Preferences
- Explain every block of code after writing it
- When there are multiple approaches, briefly state the tradeoff before choosing
- Flag when a concept is mathematically important
- Prefer clarity over cleverness

## Current Phase
PHASE 0 — Environment setup and project scaffolding

## What NOT to do
- Do not use deprecated scipy or numpy syntax
- Do not skip docstrings to save space
- Do not combine multiple responsibilities in one function
- Do not use notebooks inside src/