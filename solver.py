# --- solver.py ---

import argparse
import time
import requests
from collections import Counter
import utils
from utils import N, MAPPING, log_with_time, vlog
from board import print_board, compute_board_score
import time  # Ensure time is available in imported modules

# Ensure search module has access to time
import search
import sys
if 'time' not in search.__dict__:
    search.time = time

from search import parallel_first_beam

API_URL  = 'https://scraple.io/api/daily-puzzle'
DICT_URL = 'https://scraple.io/dictionary.txt'

def fetch_board_and_rack():
    resp = requests.get(API_URL)
    resp.raise_for_status()
    data = resp.json()
    board = [['' for _ in range(N)] for _ in range(N)]
    for bonus, (r, c) in data['bonusTilePositions'].items():
        board[r][c] = MAPPING[bonus]
    rack = [t['letter'].upper() for t in data['letters']]
    return board, rack

def load_dictionary():
    t0 = time.time()
    log_with_time("⟳ Downloading dictionary…")
    resp = requests.get(DICT_URL)
    resp.raise_for_status()
    words = [
        w.strip().upper()
        for w in resp.text.splitlines()
        if w.strip().isalpha() and 2 <= len(w.strip()) <= N
    ]
    wordset = set(words)
    vlog(f"Dictionary loaded and filtered ({len(words)} words)", t0)
    log_with_time(f"✅ {len(words)} words")
    return words, wordset

def run_solver():
    parser = argparse.ArgumentParser(description="ScrapleSolver")
    parser.add_argument('--beam-width', type=int, default=5, help='Beam width for the search (default: 5)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    utils.start_time = time.time()
    utils.VERBOSE = args.verbose

    board, rack = fetch_board_and_rack()
    original_bonus = [row[:] for row in board]
    words, wordset = load_dictionary()

    beam_width = args.beam_width
    log_with_time(f"Evaluating full {beam_width} beam width search...")
    best_total, best_results = parallel_first_beam(
        board, rack, words, wordset, original_bonus, beam_width=beam_width
    )

    if not best_results:
        log_with_time("No valid full simulation found.")
        return

    log_with_time(f"Found {len(best_results)} highest scoring solution(s) with score {best_total}:")
    for idx, (score, best_board, best_moves) in enumerate(best_results, 1):
        log_with_time(f"Solution {idx}:")
        log_with_time("Move sequence:")
        for move in best_moves:
            sc, w, d, r0, c0 = move
            log_with_time(f"  {w} at ({r0},{c0}) {d} scoring {sc}")
        log_with_time("Final simulated board:")
        print_board(best_board)
        print(f"True board score: {compute_board_score(best_board, original_bonus)}")
        print("-" * 40)

    total_elapsed = time.time() - utils.start_time
    print(f"Total time: {int(total_elapsed // 60)}m {total_elapsed % 60:.1f}s")