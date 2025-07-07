# --- utils.py ---

import time
import threading
import json
from colorama import Fore, Style, init
import os

init()

# Board dimension
N = 5

# Bonus-square codes
MAPPING = {
    'DOUBLE_LETTER': 'DL',
    'TRIPLE_LETTER': 'TL',
    'DOUBLE_WORD':   'DW',
    'TRIPLE_WORD':   'TW'
}

# Standard Scrabble letter values
LETTER_SCORES = {
    **dict.fromkeys(list("AEILNORSTU"), 1),
    **dict.fromkeys(list("DG"), 2),
    **dict.fromkeys(list("BCMP"), 3),
    **dict.fromkeys(list("FHVWY"), 4),
    'K': 5,
    **dict.fromkeys(list("JX"), 8),
    **dict.fromkeys(list("QZ"), 10)
}

VERBOSE = False
start_time = None

# Lock used for synchronized printing across threads
PRINT_LOCK = threading.Lock()

def log_with_time(msg, color=Fore.LIGHTBLUE_EX):
    """Print ``msg`` with a timestamp."""
    global start_time
    elapsed = time.time() - start_time
    mins = int(elapsed // 60)
    secs = elapsed % 60
    timestamp = Style.DIM + f"[{mins:02}:{secs:06.3f}]" + Style.RESET_ALL
    with PRINT_LOCK:
        print(f"{timestamp} {color}{msg}{Style.RESET_ALL}", flush=True)

def vlog(msg, t0=None):
    if VERBOSE:
        if t0 is not None:
            elapsed = time.time() - t0
            log_with_time(f"{msg} (took {elapsed:.3f}s)")
        else:
            log_with_time(msg)

def log_puzzle_to_file(api_response, best_result=None):
    """Log the day's puzzle (board, rack, and optionally best result) to a dated JSON file in the `logs` directory.
    If the file exists, update best_result only if the new score is higher."""
    logs_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"puzzle_{time.strftime('%Y-%m-%d')}.json")

    # Parse the API response if it's a string
    if isinstance(api_response, str):
        try:
            api_data = json.loads(api_response)
        except Exception:
            api_data = {"raw": api_response}
    else:
        api_data = api_response

    log_data = {"puzzle": api_data}

    # If file exists, load and compare best_result
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                existing = json.load(f)
            log_data = existing
        except Exception:
            pass

    if best_result:
        existing_best = log_data.get("best_result")
        if not existing_best or best_result["score"] > existing_best.get("score", float('-inf')):
            log_data["best_result"] = best_result
            log_with_time(f"Updated best_result in {log_file}", color=Fore.GREEN)
        else:
            log_with_time(f"Existing best_result in {log_file} has equal or higher score; not updated.", color=Fore.YELLOW)
    else:
        log_with_time(f"Puzzle logged to {log_file}", color=Fore.GREEN)

    # Ensure 'human_best' key exists for manual entry
    if 'human_best' not in log_data:
        log_data['human_best'] = {}

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
