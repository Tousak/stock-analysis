# Roadmap: Upgrade Core Model to XGBoost

## 🎯 Objective
Replace the current `RandomForestRegressor` baseline with `XGBRegressor` to better handle the noise and non-linear relationships of financial time series data. Implement basic Hyperparameter Tuning.

## ⚠️ Critical Coding Guidelines (MANDATORY)
*   **KISS (Keep It Simple, Stupid):** No over-engineering. Write bare-bones implementations.
*   **Human-Readable & Short:** Keep the code as concise as absolutely possible.
*   **NO Error Blocks / NO Try Blocks:** The `try...except` pattern is strictly forbidden. Assume data is correct or let the program fail loudly.

## 📂 Architecture
*   **Backend (`src/`):** Update the model training and prediction logic inside `src/model.py`.
*   **Frontend (`pages/`):** Update `pages/2_Strategy_Lab.py` to allow the user to select XGBoost parameters or visualize XGBoost-specific feature importance.

## 🚀 Implementation Steps

1.  **Add Dependencies:** Add `xgboost` (and potentially `optuna` for tuning) to the project via `uv add xgboost optuna`.
2.  **Modify `src/model.py`:**
    *   Swap `RandomForestRegressor` for `XGBRegressor`.
    *   Set strict baseline parameters to limit overfitting (e.g., `max_depth=3`, `n_estimators=100`, `learning_rate=0.05`).
3.  **Basic Hyperparameter Tuning (Optional but recommended):**
    *   Implement a short, simple Bayesian optimization function (using Optuna or Hyperopt) in `src/model.py` to find the best `max_depth`, `alpha`, and `lambda` (L1/L2 regularization).
4.  **Update Strategy Lab UI:**
    *   Modify `pages/2_Strategy_Lab.py` to reflect the new XGBoost model.
    *   Ensure the feature importance chart works correctly with the XGBoost output.
