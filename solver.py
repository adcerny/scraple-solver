# --- solver.py ---

import argparse
import time
import requests
from collections import Counter
import utils
from utils import N, MAPPING, log_with_time, vlog, LETTER_SCORES
import os
from colorama import Fore
from board import print_board, compute_board_score
import time  # Ensure time is available in imported modules
from functools import lru_cache
from score_cache import board_to_tuple, cached_board_score
import json
from datetime import datetime

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

    # Log the puzzle to a file
    board_data = resp.text  # Exact string format from API
    rack_data = ','.join(rack)
    log_puzzle_to_file(board_data, rack_data)

    return board, rack

def log_puzzle_to_file(board_data, rack_data):
    """Logs the day's puzzle (board and rack) to a file."""
    today = datetime.now().strftime('%Y-%m-%d')
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{today}-puzzle.log')

    with open(log_file, 'w') as f:
        f.write('Board:\n')
        f.write(board_data + '\n')
        f.write('Rack:\n')
        f.write(rack_data + '\n')

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
    parser.add_argument('--high-score-deep-dive', nargs='?', const=1000, type=int,
                        help='After initial search, explore all subsequent moves for the best starting word. Optionally specify beam width (default: 1000)')
    parser.add_argument('--load-log', type=str, default=None, help='Path to a JSON log file to load the puzzle from instead of calling the API')
    parser.add_argument('--start-word', type=str, default=None, help='Specify a start word to force as the first move')
    args = parser.parse_args()

    beam_width = args.beam_width
    first_moves = args.first_moves
    max_moves = args.depth

    utils.start_time = time.time()
    utils.VERBOSE = args.verbose

    # Pass cache disable flag to score_cache
    import score_cache
    score_cache.CACHE_DISABLED = args.no_cache

    if args.load_log:
        try:
            with open(args.load_log, 'r') as f:
                log_data = json.load(f)
        except FileNotFoundError:
            log_with_time(f"Could not find log file: {args.load_log}", color=Fore.RED)
            return
        except Exception as e:
            log_with_time(f"Error loading log file: {e}", color=Fore.RED)
            return
        puzzle = log_data['puzzle']
        board = [['' for _ in range(N)] for _ in range(N)]
        for bonus, pos in puzzle['bonusTilePositions'].items():
            if isinstance(pos[0], int):
                r, c = pos
                board[r][c] = MAPPING[bonus]
        rack = [t['letter'].upper() for t in puzzle['letters']]
    else:
        # Fetch the board and rack
        board, rack = fetch_board_and_rack()

    # Log the puzzle if the argument is provided
    if args.log_puzzle:
        api_response = json.dumps({
            "letters": [{"letter": t.upper(), "points": LETTER_SCORES[t.upper()]} for t in rack],
            "bonusTilePositions": {bonus: pos for bonus, pos in MAPPING.items()},
            "date": time.strftime('%Y-%m-%d'),
            "displayDate": time.strftime('%B %d, %Y')
        })
        utils.log_puzzle_to_file(api_response)

    print("Today's Board:")
    print_board(board)
    print("Rack:", ' '.join(rack))
    original_bonus = [row[:] for row in board]
    words, wordset = load_dictionary()

    # If --start-word is provided, check if it can be formed from the rack and use it as the first move
    if args.start_word:
        start_word = args.start_word.upper()
        log_with_time(f"Using start word: '{start_word}'", color=Fore.YELLOW)
        rack_counter = Counter(rack)
        word_counter = Counter(start_word)
        if any(word_counter[ch] > rack_counter.get(ch, 0) for ch in word_counter):
            log_with_time(f"Cannot form start word '{start_word}' from rack: {' '.join(rack)}", color=Fore.RED)
            return
        from search import find_best, beam_from_first, prune_words
        # Find all valid placements for the start word
        valid_placements = find_best(
            board,
            rack_counter,
            [start_word],
            wordset,
            None,
            original_bonus,
            top_k=None
        )
        if not valid_placements:
            log_with_time(f"No valid placements for start word '{start_word}' on the board.", color=Fore.RED)
            return
        best_placement = max(valid_placements, key=lambda x: x[0])
        log_with_time(f"Best placement for '{start_word}': score {best_placement[0]}, position ({best_placement[3]},{best_placement[4]}) {best_placement[2]}", color=Fore.YELLOW)
        # Remove start word letters from rack for pruning
        rack_after_first = rack_counter.copy()
        for ch in start_word:
            rack_after_first[ch] -= 1
            if rack_after_first[ch] == 0:
                del rack_after_first[ch]
        pruned_words = prune_words(words, rack_after_first, board)
        log_with_time(f"Pruned word list for subsequent moves: {len(pruned_words)} words")
        score, board_after, moves = beam_from_first(
            best_placement,
            board,
            rack_counter,
            pruned_words,
            wordset,
            original_bonus,
            beam_width=beam_width,
            max_moves=max_moves
        )
        log_with_time(f"Best result with start word '{start_word}': {score}", color=Fore.GREEN)
        log_with_time("Move sequence:", color=Fore.GREEN)
        for move in moves:
            sc, w, d, r0, c0 = move
            log_with_time(f"  {w} at ({r0},{c0}) {d} scoring {sc}", color=Fore.GREEN)
        log_with_time("Final simulated board:", color=Fore.GREEN)
        print()
        print_board(board_after, original_bonus)
        print(f"Final board score: {cached_board_score(board_to_tuple(board_after), board_to_tuple(original_bonus))}")
        print("-" * 40)
        return

    # Normal game logic (all words) only runs if --start-word is not provided

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

    if args.high_score_deep_dive and best_results:
        deep_dive_beam_width = args.high_score_deep_dive if isinstance(args.high_score_deep_dive, int) else 1000
        # Find the highest scoring first move among all best_results
        best_first_move = None
        best_first_score = float('-inf')
        for score, board_candidate, moves in best_results:
            if moves and moves[0][0] > best_first_score:
                best_first_score = moves[0][0]
                best_first_move = moves[0]

        if best_first_move:
            log_with_time(
                f"High score deep dive starting from {best_first_move[1]} at ({best_first_move[3]},{best_first_move[4]}) {best_first_move[2]}",
                color=Fore.YELLOW,
            )
            rack_count = Counter(rack)
            dive_score, dive_board, dive_moves = search.beam_from_first(
                best_first_move,
                board,
                rack_count,
                words,
                wordset,
                original_bonus,
                beam_width=deep_dive_beam_width,
                max_moves=max_moves,
            )
            if dive_board:
                log_with_time(
                    f"Deep dive best score: {dive_score}",
                    color=Fore.YELLOW,
                )
                log_with_time("Move sequence:", color=Fore.YELLOW)
                for move in dive_moves:
                    sc, w, d, r0, c0 = move
                    log_with_time(
                        f"  {w} at ({r0},{c0}) {d} scoring {sc}",
                        color=Fore.YELLOW,
                    )
                log_with_time("Final board:", color=Fore.YELLOW)
                print_board(dive_board, original_bonus)

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
