import os
import json
import openai
from dotenv import load_dotenv

# Load environment variables from a .env file in the project root
load_dotenv()

# Configure the OpenAI client
api_key = os.getenv("API_KEY_OPENAI")
if not api_key:
    raise ValueError("API_KEY_OPENAI environment variable not found. Please set it in your .env file.")

client = openai.OpenAI(api_key=api_key)

def get_openai_sentiment(mda_text: str, model: str = "gpt-4o-mini") -> dict:
    """
    Analyzes the sentiment of a given MD&A text using an OpenAI model.

    Args:
        mda_text (str): The text from the Management's Discussion & Analysis.
        model (str): The OpenAI model to use for the analysis.

    Returns:
        dict: A dictionary containing 'score' (float) and 'justification' (str).
              Returns a neutral score and error message on failure.
    """
    system_prompt = """
You are a specialized financial analyst. Your task is to analyze the sentiment of a
company's Management's Discussion and Analysis (MD&A) from an investor's perspective.
Focus on forward-looking statements, risk factors, and overall tone regarding future performance.

Provide your output in a JSON object with two keys:
1. "score": A float ranging from -1.0 (very bearish/negative) to 1.0 (very bullish/positive).
2. "justification": A brief, one-sentence explanation for your score.
"""

    user_prompt = f"Please analyze the following MD&A text:\n\n---\n{mda_text[:8000]}\n---" # Truncate to avoid excessive token usage

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,  # Set to 0 for deterministic output
        )
        
        # The response content is a JSON string, so it needs to be parsed.
        analysis = json.loads(response.choices[0].message.content)
        
        # Validate the response structure
        if "score" in analysis and "justification" in analysis:
            return {
                "score": float(analysis["score"]),
                "justification": str(analysis["justification"])
            }
        else:
            return {
                "score": 0.0,
                "justification": "Error: API response was in an invalid format."
            }

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return {"score": 0.0, "justification": f"Error: OpenAI API Error - {e}"}
    except json.JSONDecodeError:
        return {
            "score": 0.0,
            "justification": "Error: Failed to decode JSON response from API."
        }
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"score": 0.0, "justification": f"Error: An unexpected error occurred - {e}"}

if __name__ == '__main__':
    # Example usage with a sample text
    print("Running example sentiment analysis...")
    
    sample_mda = """
    Looking ahead, we are optimistic about our growth trajectory. Our recent investments in AI
    have begun to yield significant returns, and we anticipate this trend to continue.
    However, we are also mindful of the ongoing supply chain disruptions, which could pose a
    near-term challenge to our production capabilities. Overall, revenue increased by 15%
    year-over-year, beating expectations.
    """
    
    sentiment = get_openai_sentiment(sample_mda)
    
    print(f"Model: gpt-4o-mini")
    print(f"Sentiment Score: {sentiment['score']}")
    print(f"Justification: {sentiment['justification']}")

    # Example of a more bearish text
    print("\n" + "="*20 + "\n")
    print("Running bearish example...")
    
    bearish_mda = """
    We face significant headwinds from increased competition and regulatory scrutiny.
    Our flagship product line has seen a marked decrease in market share, and we project
    a decline in revenues for the upcoming fiscal year. While we are implementing cost-cutting
    measures, the outlook remains uncertain and challenging.
    """
    
    sentiment_bearish = get_openai_sentiment(bearish_mda)
    
    print(f"Model: gpt-4o-mini")
    print(f"Sentiment Score: {sentiment_bearish['score']}")
    print(f"Justification: {sentiment_bearish['justification']}")
