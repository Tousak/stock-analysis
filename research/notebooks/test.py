import pandas as pd
from transformers import pipeline
import os
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = r"research\notebooks\nvidia_news_2026-02-09.xlsx"
OUTPUT_FILE = r"research\notebooks\nvidia_news_sentiment_2026-02-09.xlsx"
MODEL_PATH = r"research\notebooks\local_finbert"

def analyze_sentiment():
    print(f"Loading data from: {INPUT_FILE}")
    try:
        df = pd.read_excel(INPUT_FILE)
        print(f"Loaded {len(df)} rows.")
    except FileNotFoundError:
        print("Error: Input file not found. Check the path.")
        return

    print(f"Loading local FinBERT model from: {MODEL_PATH}")
    try:
        # UPDATED: Use top_k=None to ensure we get ALL scores (Pos, Neg, Neu)
        # device=-1 forces CPU
        sentiment_pipeline = pipeline(
            "text-classification", 
            model=MODEL_PATH, 
            tokenizer=MODEL_PATH, 
            device=-1, 
            top_k=None 
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare lists
    sentiments = []
    confidences = []
    pos_scores = []
    neg_scores = []
    neu_scores = []

    print("Analyzing sentiment...")
    
    for text in tqdm(df['title']):
        try:
            clean_text = str(text)
            
            # Run inference
            result = sentiment_pipeline(clean_text, truncation=True, max_length=512)
            
            # --- FIX IS HERE ---
            # Check structure: Is result[0] a list (multiple scores) or a dict (single score)?
            if isinstance(result[0], list):
                # Standard format: [[{'label': 'A'...}, {'label': 'B'...}]]
                predictions = result[0]
            else:
                # Flat format: [{'label': 'A'...}, {'label': 'B'...}]
                predictions = result

            # Extract scores
            score_dict = {item['label']: item['score'] for item in predictions}
            
            # Determine winner
            best_label = max(score_dict, key=score_dict.get)
            
            sentiments.append(best_label)
            confidences.append(score_dict[best_label])
            pos_scores.append(score_dict.get('positive', 0))
            neg_scores.append(score_dict.get('negative', 0))
            neu_scores.append(score_dict.get('neutral', 0))
            
        except Exception as e:
            # If it still fails, log it but don't crash the whole script
            print(f"\nSkipping row: '{str(text)[:20]}...' | Error: {e}")
            sentiments.append("error")
            confidences.append(0.0)
            pos_scores.append(0.0)
            neg_scores.append(0.0)
            neu_scores.append(0.0)

    # Save results
    df['sentiment'] = sentiments
    df['sentiment_confidence'] = confidences
    df['score_positive'] = pos_scores
    df['score_negative'] = neg_scores
    df['score_neutral'] = neu_scores

    print(f"Saving results to: {OUTPUT_FILE}")
    df.to_excel(OUTPUT_FILE, index=False)
    print("Done! Analysis complete.")

if __name__ == "__main__":
    analyze_sentiment()