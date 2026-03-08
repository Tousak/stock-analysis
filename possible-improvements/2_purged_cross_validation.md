# Roadmap: Combinatorial Purged Cross-Validation (CPCV)

## 🎯 Objective
Replace the naive time-based walk-forward split with Purged K-Fold or Combinatorial Purged Cross-Validation (CPCV) to eliminate data leakage and ensure backtest robustness.

## ⚠️ Critical Coding Guidelines (MANDATORY)
*   **KISS (Keep It Simple, Stupid):** No over-engineering. Write bare-bones implementations.
*   **Human-Readable & Short:** Keep the code as concise as absolutely possible.
*   **NO Error Blocks / NO Try Blocks:** The `try...except` pattern is strictly forbidden. Assume data is correct or let the program fail loudly.

## 📂 Architecture
*   **Backend (`src/`):** Update `src/model.py` and `src/backtester.py`.
*   **Frontend (`pages/`):** Update `pages/2_Strategy_Lab.py` to reflect the new cross-validation strategy.

## 🚀 Implementation Steps

1.  **Implement Purging Logic:**
    *   In `src/model.py`, when creating the split for training and testing, implement strict "purging".
    *   Find the exact start date of the Test set.
    *   Remove any rows from the Training set where `filing_date + 90 days` (the holding period) overlaps with or exceeds the start of the Test set.
2.  **Implement K-Fold (Optional CPCV):**
    *   Instead of a single walk-forward train/test split, implement a simple Purged K-Fold cross-validation over the timeline.
    *   Calculate the model's accuracy (MSE) over *all* folds, rather than one final test set, to get a true representation of the model's robustness.
3.  **Update Strategy Lab UI:**
    *   Modify `pages/2_Strategy_Lab.py` to display the cross-validated MSE and performance across multiple historical paths, rather than a single trajectory.
