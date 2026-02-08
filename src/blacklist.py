import os
from src.config import BLACKLIST_PATH

def get_blacklist() -> set:
    """
    Loads the blacklist of tickers from the specified file.

    Returns:
        set: A set of blacklisted ticker symbols. Returns an empty set if the file doesn't exist.
    """
    if not os.path.exists(BLACKLIST_PATH):
        return set()
    try:
        with open(BLACKLIST_PATH, 'r') as f:
            # Read lines, strip whitespace, and filter out empty lines
            return {line.strip().upper() for line in f if line.strip()}
    except Exception as e:
        print(f"Warning: Could not read blacklist file at {BLACKLIST_PATH}. Error: {e}")
        return set()

def add_to_blacklist(tickers: list[str]):
    """
    Adds a list of tickers to the blacklist file.

    Args:
        tickers (list[str]): A list of ticker symbols to add.
    """
    if not tickers:
        return

    # Ensure the directory exists
    os.makedirs(os.path.dirname(BLACKLIST_PATH), exist_ok=True)
    
    current_blacklist = get_blacklist()
    new_tickers = {ticker.upper() for ticker in tickers if ticker} # Normalize to uppercase
    
    # Only add tickers that are not already on the list
    tickers_to_add = new_tickers - current_blacklist
    
    if not tickers_to_add:
        print("No new tickers to add to the blacklist.")
        return

    try:
        with open(BLACKLIST_PATH, 'a') as f:
            for ticker in tickers_to_add:
                f.write(f"{ticker}\n")
        print(f"Added {len(tickers_to_add)} ticker(s) to the blacklist: {', '.join(tickers_to_add)}")
    except Exception as e:
        print(f"Error: Could not write to blacklist file at {BLACKLIST_PATH}. Error: {e}")

def is_blacklisted(ticker: str) -> bool:
    """
    Checks if a single ticker is in the blacklist.

    Args:
        ticker (str): The ticker symbol to check.

    Returns:
        bool: True if the ticker is in the blacklist, False otherwise.
    """
    return ticker.upper() in get_blacklist()

if __name__ == "__main__":
    print("Running blacklist.py example...")
    
    # Clean up previous test runs if file exists
    if os.path.exists(BLACKLIST_PATH):
        os.remove(BLACKLIST_PATH)
        print(f"Removed existing blacklist file for fresh test.")

    # --- Test 1: Load empty blacklist ---
    blacklist = get_blacklist()
    print(f"Initial blacklist: {blacklist}")
    assert len(blacklist) == 0

    # --- Test 2: Add tickers ---
    add_to_blacklist(['BAD1', 'BAD2'])
    blacklist = get_blacklist()
    print(f"Blacklist after adding: {blacklist}")
    assert 'BAD1' in blacklist and 'BAD2' in blacklist

    # --- Test 3: Check if tickers are blacklisted ---
    print(f"Is 'BAD1' blacklisted? {is_blacklisted('BAD1')}")
    assert is_blacklisted('BAD1') == True
    print(f"Is 'GOOD' blacklisted? {is_blacklisted('GOOD')}")
    assert is_blacklisted('GOOD') == False

    # --- Test 4: Add existing and new tickers ---
    print("Adding 'BAD2' (existing) and 'BAD3' (new)...")
    add_to_blacklist(['bad2', 'BAD3']) # Test lowercase
    blacklist = get_blacklist()
    print(f"Blacklist after adding duplicates/new: {blacklist}")
    assert len(blacklist) == 3

    # --- Test 5: Clean up ---
    if os.path.exists(BLACKLIST_PATH):
        os.remove(BLACKLIST_PATH)
        print(f"Test complete. Cleaned up blacklist file.")
