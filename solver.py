# --- solver.py ---

import argparse
import time
import requests
from collections import Counter
import utils
from utils import N, MAPPING, log_with_time, vlog
from colorama import Fore
from board import print_board, compute_board_score
import time  # Ensure time is available in imported modules
from functools import lru_cache
from score_cache import board_to_tuple, cached_board_score
import json

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
    parser.add_argument('--beam-width', type=int, default=10, help='Beam width for the search (default: 10)')
    parser.add_argument('--first-moves', type=int, default=None, help='Number of opening moves to explore (default: beam width)')
    parser.add_argument('--depth', type=int, default=20, help='Maximum number of moves to search (default: 20)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-cache', action='store_true', help='Disable board score caching')
    parser.add_argument('--log-puzzle', action='store_true', help='Save the day\'s puzzle and best result to a JSON log file')
    args = parser.parse_args()

    utils.start_time = time.time()
    utils.VERBOSE = args.verbose

    # Pass cache disable flag to score_cache
    import score_cache
    score_cache.CACHE_DISABLED = args.no_cache

    board, rack = fetch_board_and_rack()
    print("Today's Board:")
    print_board(board)
    print("Rack:", ' '.join(rack))
    original_bonus = [row[:] for row in board]
    words, wordset = load_dictionary()

    # Save the API response for logging
    api_response = None
    if args.log_puzzle:
        # Fetch the raw API response for logging
        resp = requests.get(API_URL)
        resp.raise_for_status()
        api_response = resp.text

    beam_width = args.beam_width
    first_moves = args.first_moves
    max_moves = args.depth
    log_with_time(
        f"Evaluating full {beam_width} beam width search with {first_moves or beam_width} first moves and max depth {max_moves}..."
    )

    # Run the search and collect best results as they are found
    best_total, best_results = parallel_first_beam(
        board,
        rack,
        words,
        wordset,
        original_bonus,
        beam_width=beam_width,
        first_moves=first_moves,
        max_moves=max_moves,
    )

    if not best_results:
        log_with_time("No valid full simulation found.")
        return

    log_with_time(
        f"Found {len(best_results)} highest scoring solution(s) with score {best_total}:",
        color=Fore.GREEN,
    )
    unique_count = 0
    for idx, (score, best_board, best_moves) in enumerate(best_results, 1):
        unique_count += 1
        log_with_time(f"Solution {unique_count}:", color=Fore.GREEN)
        log_with_time("Move sequence:", color=Fore.GREEN)
        for move in best_moves:
            sc, w, d, r0, c0 = move
            log_with_time(f"  {w} at ({r0},{c0}) {d} scoring {sc}", color=Fore.GREEN)
        log_with_time("Final simulated board:", color=Fore.GREEN)
        print()
        print_board(best_board, original_bonus)
        print(f"Final board score: {cached_board_score(board_to_tuple(best_board), board_to_tuple(original_bonus))}")
        print("-" * 40)

    # Log the puzzle and best result if requested
    if args.log_puzzle and best_results:
        best_score, best_board, _ = best_results[0]
        best_result = {
            "score": best_score,
            "final_board": best_board
        }
        utils.log_puzzle_to_file(api_response, best_result=best_result)

    total_elapsed = time.time() - utils.start_time
    print(f"Total time: {int(total_elapsed // 60)}m {total_elapsed % 60:.1f}s")
