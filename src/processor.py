import json
import pandas as pd
from tqdm.auto import tqdm
import openai

from src.config import client

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

def process_filings_for_sentiment(filings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame of raw filings to analyze the sentiment of the 'mda_text' column.
    """
    if 'mda_text' not in filings_df.columns:
        raise ValueError("Input DataFrame must contain an 'mda_text' column.")

    print("Analyzing sentiment for new filings...")
    tqdm.pandas(desc="Analyzing Sentiment")
    sentiment_results = filings_df['mda_text'].progress_apply(get_sentiment_openai)
    
    filings_df['sentiment_score'] = sentiment_results.apply(lambda x: x.get('score', 0.0))
    filings_df['sentiment_justification'] = sentiment_results.apply(lambda x: x.get('justification', ''))
    
    return filings_df

