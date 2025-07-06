# --- utils.py ---

import time
import threading
from colorama import Fore, Style, init

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

# Predefined colors for rainbow text
RAINBOW_COLORS = [
    Fore.RED,
    Fore.YELLOW,
    Fore.GREEN,
    Fore.CYAN,
    Fore.BLUE,
    Fore.MAGENTA,
]

# Lock used for synchronized printing across threads
PRINT_LOCK = threading.Lock()

def rainbow(text):
    """Return text colored sequentially with rainbow colors."""
    out = []
    for i, ch in enumerate(text):
        color = RAINBOW_COLORS[i % len(RAINBOW_COLORS)]
        out.append(color + ch)
    out.append(Style.RESET_ALL)
    return ''.join(out)

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
