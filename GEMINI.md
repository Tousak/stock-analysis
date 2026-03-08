# Project Context: Stock Analysis Dashboard

This document provides essential context and instructions for AI agents maintaining or extending this repository.

## Project Overview
This project is a stock analysis application that predicts stock prices based on annual reports fetched using the `edgar` library. 
The application features a frontend dashboard built with Streamlit and a backend pipeline for data fetching, sentiment analysis, feature engineering, and predictive modeling.

## Tech Stack
- **Frontend**: Streamlit (`pages/` directory and `app.py`)
- **Backend Pipeline**: Python (`src/` directory and `main.py`)
- **Package Management**: **UV** (Do not use `pip` for installing packages; rely on `uv`)

## Architecture
- `app.py`: High-level executive summary and Streamlit entry point.
- `main.py`: CLI entry point for running the data pipeline.
- `src/`: Backend logic containing the following components:
  - `config.py`: Configuration and constants.
  - `data_loader.py`: Fetches SEC filings.
  - `processor.py`: Sentiment analysis processing.
  - `feature_eng.py`: Feature engineering.
  - `model.py`: Model training and predictions.
  - `backtester.py`: Portfolio simulation.
- `pages/`: Streamlit frontend pages.

## Critical Coding Guidelines

Whenever generating new code or modifying existing code in this repository, you **MUST** adhere to the following strict rules:

1. **Keep It Simple, Stupid (KISS)**
   - Prioritize straightforward, bare-bones implementations.
   - Avoid over-engineering, unnecessary abstractions, or complex design patterns.
   
2. **Human-Readable and Short**
   - Write clean code that is easy for a human to read at a glance.
   - Keep functions and scripts as short as absolutely possible while maintaining correct logic.

3. **NO Error Blocks / NO Try Blocks**
   - Under no circumstances is the `try...except` pattern allowed in new code.
   - Assume the data is correct or allow the program to fail loudly if something is completely broken. Do not wrap code in error-handling blocks.

**Failure to follow these guidelines is completely unacceptable.**
