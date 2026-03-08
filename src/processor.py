import json
import pandas as pd
from tqdm.auto import tqdm
import openai
from transformers import pipeline

from src.config import client

# Initialize the FinBERT pipeline globally so it's only loaded once when the module imports.
try:
    finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception as e:
    finbert_pipeline = None
    print(f"Warning: Could not load FinBERT - {e}")

def get_sentiment_openai(mda_text: str, model: str = "gpt-4o-mini") -> dict:
    """
    Analyzes the sentiment of a given MD&A text using the pre-configured OpenAI client.
    (Error handling removed for debugging).
    """
    if not mda_text or not isinstance(mda_text, str) or len(mda_text) < 50:
        return {"score": 0.0, "justification": "No valid MD&A text provided to analyze."}

    system_prompt = """
    You are a specialized financial analyst. Your task is to analyze the sentiment of a
    company's Management's Discussion and Analysis (MD&A) from an investor's perspective.
    Focus on forward-looking statements, risk factors, and overall tone regarding future performance.

    Provide your output in a JSON object with two keys:
    1. "score": A float ranging from -1.0 (very bearish/negative) to 1.0 (very bullish/positive).
    2. "justification": A brief, one-sentence explanation for your score.
    """
    user_prompt = f"Please analyze the following MD&A text:\n\n---\n{mda_text[:8000]}\n---" 

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        
        analysis = json.loads(response.choices[0].message.content)
        
        if "score" in analysis and "justification" in analysis:
            return {
                "score": float(analysis["score"]),
                "justification": str(analysis["justification"])
            }
        else:
            return {"score": 0.0, "justification": "Error: API response was in an invalid format."}

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return {"score": 0.0, "justification": f"Error: OpenAI API Error - {e}"}
    except Exception as e:
        print(f"An unexpected error occurred during sentiment analysis: {e}")
        return {"score": 0.0, "justification": f"Error: An unexpected error occurred - {e}"}

def get_sentiment_finbert(mda_text: str) -> dict:
    """
    Analyzes sentiment using the local HuggingFace ProsusAI/finbert model.
    Chunks the text to respect the 512 token limit and averages the probabilities.
    Returns scalar score + triplet (pos, neg, neu probabilities).
    """
    if not mda_text or not isinstance(mda_text, str) or len(mda_text) < 50:
        return {"score": 0.0, "pos": 0.33, "neg": 0.33, "neu": 0.33, "justification": "No valid MD&A text provided."}
    
    if finbert_pipeline is None:
        return {"score": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "justification": "FinBERT model failed to load."}

    # KISS chunking: Split by roughly 400 words to stay under 512 tokens
    words = mda_text.split()
    chunk_size = 400
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    
    total_pos, total_neg, total_neu = 0.0, 0.0, 0.0
    valid_chunks = 0
    
    for chunk in chunks:
        try:
            # top_k=None returns all scores: [{'label': 'positive', 'score': ...}, {'label': 'negative', 'score': ...}, {'label': 'neutral', 'score': ...}]
            results = finbert_pipeline(chunk, top_k=None)
            
            # Map labels to scores
            scores = {r['label']: r['score'] for r in results}
            
            total_pos += scores.get('positive', 0.0)
            total_neg += scores.get('negative', 0.0)
            total_neu += scores.get('neutral', 0.0)
            valid_chunks += 1
        except:
            continue
            
    if valid_chunks == 0:
        return {"score": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "justification": "Failed to process any text chunks."}
        
    avg_pos = total_pos / valid_chunks
    avg_neg = total_neg / valid_chunks
    avg_neu = total_neu / valid_chunks
    
    # Combined scalar score (-1 to 1)
    scalar_score = avg_pos - avg_neg
    
    return {
        "score": scalar_score, 
        "pos": avg_pos, 
        "neg": avg_neg, 
        "neu": avg_neu,
        "justification": f"Average FinBERT scores over {valid_chunks} chunks."
    }


def process_filings_for_sentiment(filings_df: pd.DataFrame, nlp_method: str = "finbert") -> pd.DataFrame:
    """
    Processes a DataFrame of raw filings to analyze the sentiment of the 'mda_text' column.
    Allows selecting between 'finbert' (local, leak-free) and 'openai'.
    """
    if 'mda_text' not in filings_df.columns:
        raise ValueError("Input DataFrame must contain an 'mda_text' column.")

    print(f"Analyzing sentiment for new filings using {nlp_method}...")
    tqdm.pandas(desc=f"Analyzing Sentiment ({nlp_method})")
    
    if nlp_method == "openai":
        sentiment_results = filings_df['mda_text'].progress_apply(get_sentiment_openai)
    else:
        sentiment_results = filings_df['mda_text'].progress_apply(get_sentiment_finbert)
    
    filings_df['sentiment_score'] = sentiment_results.apply(lambda x: x.get('score', 0.0))
    filings_df['sentiment_pos'] = sentiment_results.apply(lambda x: x.get('pos', 0.0))
    filings_df['sentiment_neg'] = sentiment_results.apply(lambda x: x.get('neg', 0.0))
    filings_df['sentiment_neu'] = sentiment_results.apply(lambda x: x.get('neu', 0.0))
    filings_df['sentiment_justification'] = sentiment_results.apply(lambda x: x.get('justification', ''))
    
    return filings_df

