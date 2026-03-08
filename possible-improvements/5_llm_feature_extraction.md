# Roadmap: Advanced LLM Feature Extraction (Chain-of-Thought)

## 🎯 Objective
Upgrade the `gpt-4o-mini` natural language processor from a basic sentiment oracle to a sophisticated feature extractor using Chain-of-Thought (CoT) reasoning. Emphasize the calculation of the "Sentiment Delta" over a trailing window.

## ⚠️ Critical Coding Guidelines (MANDATORY)
*   **KISS (Keep It Simple, Stupid):** No over-engineering. Write bare-bones implementations.
*   **Human-Readable & Short:** Keep the code as concise as absolutely possible.
*   **NO Error Blocks / NO Try Blocks:** The `try...except` pattern is strictly forbidden. Assume data is correct or let the program fail loudly.

## 📂 Architecture
*   **Backend (`src/`):** Update the prompt and extraction logic in `src/processor.py` for either 10-Q SEC data or daily News data. Update `src/feature_eng.py` to prioritize the sentiment delta calculation.

## 🚀 Implementation Steps

1.  **Enhance LLM Prompting:**
    *   In `src/processor.py`, upgrade the `system_prompt` and `user_prompt` passed to OpenAI to explicitly require step-by-step reasoning (e.g., "Analyze the text for forward-looking risks, then rate the sentiment from -1.0 to 1.0").
    *   Ensure the JSON output guarantees a reliable schema (e.g., `{"reasoning": "...", "score": 0.5}`).
2.  **Calculate Sentiment Delta (Acceleration):**
    *   In `src/feature_eng.py`, instead of relying heavily on the absolute `sentiment_score`, create a robust, trailing average calculation for `sentiment_change`.
    *   The model must calculate the difference between the *current* sentiment score and the *lagging 5-day* (or previous quarter's) sentiment score to measure the "acceleration" or "deceleration" of market perception.
3.  **Local NLP Option (Future consideration):**
    *   If OpenAI API costs become restrictive due to processing daily high-frequency news, write a secondary pipeline using the local HuggingFace `ProsusAI/finbert` model as an alternative to `gpt-4o-mini`.
