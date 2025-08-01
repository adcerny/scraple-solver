def print_leaderboard_summary(best_score, leaderboard_data):
    leaderboard_scores = [entry["score"] for entry in leaderboard_data.get("scores", [])]
    if leaderboard_scores:
        rank = 1 + sum(1 for s in leaderboard_scores if s > best_score)
        high_score = max(leaderboard_scores)
        if best_score < high_score:
            print(Fore.LIGHTYELLOW_EX + f"\nYour best score ({best_score}) would rank: {rank} out of {len(leaderboard_scores)} on the current leaderboard.")
            diff = high_score - best_score
            print(Fore.LIGHTYELLOW_EX + f"Your score is {diff} points lower than the current leaderboard high score of {high_score}")
            print(Fore.RESET, end="")
        elif best_score == high_score:
            print(Fore.CYAN + f"\nYour best score ({best_score}) matches the current leaderboard high score!")
            print(Fore.CYAN + f"You are tied for the high score! Rank: {rank} out of {len(leaderboard_scores)}")
            print(Fore.RESET, end="")
        else:
            diff = best_score - high_score
            print(Fore.GREEN + f"\nCongratulations! Your score {best_score} is {diff} higher than the current high score of {high_score}")
            print(Fore.GREEN + f"Your score would be #1 on the leaderboard! Rank: {rank} out of {len(leaderboard_scores)}")
            print(Fore.RESET, end="")
    else:
        print("Could not parse leaderboard scores.")

import argparse
import time
import requests
from collections import Counter
import utils
from utils import N, MAPPING, log_with_time, vlog, LETTER_SCORES, Direction
import os
from colorama import Fore
from board import print_board, compute_board_score, get_letter_mask, score_word
import time  # Ensure time is available in imported modules
from functools import lru_cache
from score_cache import board_to_tuple, cached_board_score
import json
from datetime import datetime
import concurrent.futures

# Ensure search module has access to time
import search
import sys
if 'time' not in search.__dict__:
    search.time = time

from search import parallel_first_beam

API_URL  = 'https://scraple.io/api/daily-puzzle'
DICT_URL = 'https://scraple.io/dictionary.txt'
LEADERBOARD_URL = 'https://scraple.io/api/leaderboard'


def extract_words_with_scores(board, bonus):
    """Return a list of (score, word, positions) for all words on ``board``."""
    words = []
    mask = get_letter_mask(board)
    for r in range(N):
        c = 0
        while c < N:
            if mask[r][c]:
                start = c
                while c < N and mask[r][c]:
                    c += 1
                if c - start >= 2:
                    letters = board[r][start:c]
                    bonuses = bonus[r][start:c]
                    positions = [(r, cc) for cc in range(start, c)]
                    words.append((score_word(letters, bonuses), ''.join(letters), positions))
            else:
                c += 1
    for c in range(N):
        r = 0
        while r < N:
            if mask[r][c]:
                start = r
                while r < N and mask[r][c]:
                    r += 1
                if r - start >= 2:
                    letters = [board[i][c] for i in range(start, r)]
                    bonuses = [bonus[i][c] for i in range(start, r)]
                    positions = [(i, c) for i in range(start, r)]
                    words.append((score_word(letters, bonuses), ''.join(letters), positions))
            else:
                r += 1
    return words


def remove_word_from_board(board, bonus, rack, positions, usage):
    """Remove letters for a word while tracking overlaps.

    Only clear a tile when it is no longer part of any remaining word.
    Letters cleared from the board are appended back to ``rack``.
    ``usage`` maps ``(row, col)`` to the number of words that use that tile."""
    for r, c in positions:
        if (r, c) not in usage:
            continue
        letter = board[r][c]
        usage[(r, c)] -= 1
        if usage[(r, c)] == 0:
            if letter:
                rack.append(letter)
            board[r][c] = bonus[r][c] if bonus[r][c] else ''
            del usage[(r, c)]

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

    return board, rack, data

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
    return words, wordset, resp.text

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
    parser.add_argument('--start-pos', type=str, default=None, help='Specify position and direction for start word as "row,col,dir" (e.g. "7,7,A"). Only valid if --start-word is provided.')
    parser.add_argument('--num-games', type=int, default=50, help='Number of games to play in parallel (default: 50)')
    parser.add_argument('--improve-leaderboard', action='store_true',
                        help='Start search from the current leaderboard high-score board')
    args = parser.parse_args()

    beam_width = args.beam_width
    first_moves = args.first_moves
    max_moves = args.depth
    num_games = args.num_games

    utils.start_time = time.time()
    utils.VERBOSE = args.verbose
    improvement_done = False
    best_total = None
    best_results = None

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
        leaderboard_data = None
        words, wordset = load_dictionary()[:2]
    else:
        # Fetch the board, dictionary, and leaderboard in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_board = executor.submit(fetch_board_and_rack)
            future_dict = executor.submit(load_dictionary)
            future_leaderboard = executor.submit(lambda: requests.get(LEADERBOARD_URL, timeout=10))
            board, rack, board_api_data = future_board.result()
            words, wordset, dict_text = future_dict.result()
            leaderboard_resp = future_leaderboard.result()
            try:
                leaderboard_resp.raise_for_status()
                leaderboard_data = leaderboard_resp.json()
            except Exception:
                leaderboard_data = None

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

    # Show leaderboard high score after today's board
    if not args.load_log and leaderboard_data:
        leaderboard_scores = [entry["score"] for entry in leaderboard_data.get("scores", [])]
        if leaderboard_scores:
            high_score = max(leaderboard_scores)
            highscore_entries = [entry for entry in leaderboard_data.get("scores", []) if entry["score"] == high_score]
            print(Fore.LIGHTYELLOW_EX + f"\nCurrent High Score Board Layout (Score: {high_score}):")
            from board import leaderboard_gamestate_to_board, print_board as print_board_func
            board_hs = bonus_hs = None
            game_state_hs = None
            for entry in highscore_entries:
                game_state = entry.get("gameState")
                if game_state:
                    try:
                        board_hs, bonus_hs = leaderboard_gamestate_to_board(game_state)
                        if game_state_hs is None:
                            game_state_hs = game_state
                        print_board_func(board_hs, bonus_hs)
                    except Exception as e:
                        print(Fore.LIGHTYELLOW_EX + f"    (Could not display board: {e})")
            print(Fore.RESET, end="")

            if args.improve_leaderboard and board_hs is not None:
                board = [row[:] for row in board_hs]
                original_bonus = bonus_hs
                rack = []
                if game_state_hs:
                    remaining = (
                        game_state_hs.get('rack') or
                        game_state_hs.get('remainingTiles') or
                        game_state_hs.get('letters')
                    )
                    if remaining:
                        items = remaining.values() if isinstance(remaining, dict) else remaining
                        for item in items:
                            if isinstance(item, dict):
                                letter = item.get('letter')
                                if letter:
                                    rack.append(letter.upper())
                            elif isinstance(item, str):
                                if len(item) == 1:
                                    rack.append(item.upper())
                                else:
                                    rack.extend(ch.upper() for ch in item)
                words_on_board = extract_words_with_scores(board, original_bonus)
                words_on_board.sort(key=lambda x: x[0])
                usage = {}
                for _, _, pos_list in words_on_board:
                    for pos in pos_list:
                        usage[pos] = usage.get(pos, 0) + 1
                improved = False
                for _, word_text, positions in words_on_board:
                    remove_word_from_board(board, original_bonus, rack, positions, usage)
                    new_score, new_results = parallel_first_beam(
                        board,
                        rack,
                        words,
                        wordset,
                        original_bonus,
                        beam_width=beam_width,
                        num_games=num_games,
                        first_moves=first_moves,
                        max_moves=max_moves,
                    )
                    improvement_done = True
                    if new_score > high_score:
                        best_total = new_score
                        best_results = new_results
                        improved = True
                        break
                if not improved:
                    best_total, best_results = parallel_first_beam(
                        board,
                        rack,
                        words,
                        wordset,
                        original_bonus,
                        beam_width=beam_width,
                        num_games=num_games,
                        first_moves=first_moves,
                        max_moves=max_moves,
                    )
                    improvement_done = True
    # words, wordset already loaded above

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
        pruned_words = prune_words(words, rack_counter, board)
        log_with_time(f"Pruned word list: {len(pruned_words)} words", color=Fore.CYAN)
        # If position is specified, validate and use it
        placement = None
        if args.start_pos:
            try:
                row, col, dirn = args.start_pos.split(',')
                row = int(row)
                col = int(col)
                dirn = dirn.upper()
                dir_enum = Direction.ACROSS if dirn == Direction.ACROSS.value else Direction.DOWN
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
                # Find placement matching user input
                for p in valid_placements:
                    # p = (score, word, dir, row, col, ...)
                    if p[2] == dir_enum and p[3] == row and p[4] == col:
                        placement = p
                        break
                if not placement:
                    log_with_time(f"Cannot place start word '{start_word}' at {row},{col},{dirn}.", color=Fore.RED)
                    return
            except Exception as e:
                log_with_time(f"Invalid --start-word-pos format. Use row,col,dir (e.g. 7,7,A). Error: {e}", color=Fore.RED)
                return
        else:
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
            placement = max(valid_placements, key=lambda x: x[0])
        log_with_time(
            f"Best placement for '{start_word}': score {placement[0]}, position {placement[3]},{placement[4]},{placement[2].value}",
            color=Fore.YELLOW,
        )
        rack_after_first = rack_counter.copy()
        for ch in start_word:
            rack_after_first[ch] -= 1
            if rack_after_first[ch] == 0:
                del rack_after_first[ch]
        score, board_after, moves = beam_from_first(
            placement,
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
            log_with_time(
                f"  {w} at {r0},{c0},{d.value} scoring {sc}",
                color=Fore.GREEN,
            )
        log_with_time("Final simulated board:", color=Fore.GREEN)
        print()
        print_board(board_after, original_bonus)
        print(f"Final board score: {cached_board_score(board_to_tuple(board_after), board_to_tuple(original_bonus))}")
        print("-" * 40)
        if not args.load_log and leaderboard_data:
            print_leaderboard_summary(score, leaderboard_data)
        return

    if not improvement_done:
        best_total, best_results = parallel_first_beam(
            board,
            rack,
            words,
            wordset,
            original_bonus,
            beam_width=beam_width,
            num_games=num_games,
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
            log_with_time(
                f"  {w} at {r0},{c0},{d.value} scoring {sc}",
                color=Fore.GREEN,
            )
        log_with_time("Final simulated board:", color=Fore.GREEN)
        print()
        print_board(best_board, original_bonus)
        print(f"Final board score: {cached_board_score(board_to_tuple(best_board), board_to_tuple(original_bonus))}")
        print("-" * 40)


    # At the end, print a summary of how the user's best score compares to the leaderboard high score (no board layout)
    if not args.load_log and leaderboard_data and best_results:
        best_score = best_results[0][0]
        print_leaderboard_summary(best_score, leaderboard_data)

    if args.high_score_deep_dive and best_results:
        deep_dive_beam_width = args.high_score_deep_dive if isinstance(args.high_score_deep_dive, int) else 1000
        best_first_move = None
        best_first_score = float('-inf')
        for score, board_candidate, moves in best_results:
            if moves and moves[0][0] > best_first_score:
                best_first_score = moves[0][0]
                best_first_move = moves[0]

        if best_first_move:
            log_with_time(
                f"High score deep dive starting from {best_first_move[1]} at {best_first_move[3]},{best_first_move[4]},{best_first_move[2].value}",
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
                        f"  {w} at {r0},{c0},{d.value} scoring {sc}",
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
