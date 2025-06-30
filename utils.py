# --- utils.py ---

import time

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

def log_with_time(msg):
    global start_time
    elapsed = time.time() - start_time
    mins = int(elapsed // 60)
    secs = elapsed % 60
    print(f"[{mins:02}:{secs:06.3f}] {msg}")

def vlog(msg, t0=None):
    if VERBOSE:
        if t0 is not None:
            elapsed = time.time() - t0
            log_with_time(f"{msg} (took {elapsed:.3f}s)")
        else:
            log_with_time(msg)
