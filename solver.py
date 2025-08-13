import argparse
import time
import requests
from collections import Counter
import utils
from utils import N, MAPPING, log_with_time, vlog, LETTER_SCORES, Direction
import os
from colorama import Fore, Style
from board import print_board, compute_board_score, get_letter_mask, score_word
from score_cache import board_to_tuple, cached_board_score
import json
from datetime import datetime
import concurrent.futures

# Candidate generation / beam search
from search import parallel_first_beam, beam_from_first, find_best

# -------------------------------
# External resources (current)
# -------------------------------
API_URL = "https://scraple.io/api/daily-puzzle"
LEADERBOARD_URL = "https://scraple.io/api/leaderboard"
DICT_URL = "https://scraple.io/dictionary.txt"

# -------------------------------
# DAWG hook (optional)
# -------------------------------
try:
    from dawg import DAWG
except Exception:
    DAWG = None


# -------------------------------
# Dictionary
# -------------------------------
def _build_prefix_set(words):
    prefixes = set()
    for w in words:
        for i in range(1, len(w)):
            prefixes.add(w[:i])
    return prefixes


def load_dictionary():
    """
    Download and filter a dictionary for the current board size.
    Returns (words, wordset, prefixset, dict_text).
    """
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
    prefixset = _build_prefix_set(words)
    vlog(f"Dictionary loaded and filtered ({len(words)} words)", t0)
    log_with_time(f"✅ {len(words)} words")
    return words, wordset, prefixset, resp.text


# -------------------------------
# Puzzle fetch + logging
# -------------------------------
def log_puzzle_to_file(board_data_text, rack_csv):
    """Logs the day's puzzle (raw board JSON text and rack csv) to a file."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{today}-puzzle.log")

    log_data = {
        "puzzle": json.loads(board_data_text),
        "rack": rack_csv,
        "best_result": None,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    log_with_time(f"Puzzle logged to {log_file}", color=Fore.GREEN)


def fetch_board_and_rack():
    """
    Returns:
      board (NxN, bonuses in cells as 'DL','TL','DW','TW' else '')
      rack (list of letters)
      raw_json (dict) - the API response JSON
    """
    resp = requests.get(API_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Build empty board and drop bonuses
    board = [["" for _ in range(N)] for _ in range(N)]
    for bonus, pos in data.get("bonusTilePositions", {}).items():
        # API can return either [r,c] or {"r":..,"c":..}; normalize
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            r, c = pos
        else:
            r, c = pos["row"], pos["col"]
        board[r][c] = MAPPING[bonus]

    rack = [t["letter"].upper() for t in data["letters"]]

    # Log the puzzle for reproducibility
    log_puzzle_to_file(resp.text, ",".join(rack))
    return board, rack, data


# -------------------------------
# Leaderboard utilities
# -------------------------------
def parse_leaderboard(resp):
    try:
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def leaderboard_gamestate_to_board(game_state):
    """
    Mirror of board.leaderboard_gamestate_to_board but local to solver to keep this file self-contained.
    Convert a leaderboard API gameState dict to a 2D board (list of lists of str) and a bonus board.
    """
    # Lazy import to avoid circulars
    from board import leaderboard_gamestate_to_board as _conv
    return _conv(game_state)


def print_leaderboard_summary(best_score, leaderboard_data):
    """
    Compare our best score to the leaderboard. Print a friendly summary.
    """
    if not leaderboard_data:
        print("No leaderboard data to compare.")
        return

    leaderboard_scores = [entry.get("score") for entry in leaderboard_data.get("scores", []) if "score" in entry]
    if not leaderboard_scores:
        print("Could not parse leaderboard scores.")
        return

    high_score = max(leaderboard_scores)
    # Rank where higher is better
    sorted_scores = sorted(leaderboard_scores, reverse=True)
    rank = 1
    for i, s in enumerate(sorted_scores, 1):
        if s == high_score:
            pass
        if best_score < s:
            rank = i + 1
    total = len(sorted_scores)

    if best_score < high_score:
        diff = high_score - best_score
        print(
            Fore.LIGHTYELLOW_EX
            + f"\nYour best score ({best_score}) would rank: {rank} out of {total} on the current leaderboard."
        )
        print(
            Fore.LIGHTYELLOW_EX
            + f"Your score is {diff} points lower than the current leaderboard high score of {high_score}"
        )
        print(Fore.RESET, end="")
    elif best_score == high_score:
        print(
            Fore.CYAN + f"\nYour best score ({best_score}) matches the current leaderboard high score!"
        )
        print(Fore.CYAN + f"You are tied for the high score! Rank: {rank} out of {total}")
        print(Fore.RESET, end="")
    else:
        diff = best_score - high_score
        print(
            Fore.GREEN
            + f"\nCongratulations! Your score {best_score} is {diff} higher than the current high score of {high_score}"
        )
        print(Fore.GREEN + f"Your score would be #1 on the leaderboard! Rank: {rank} out of {total}")
        print(Fore.RESET, end="")


# -------------------------------
# Board pretty-print helpers
# -------------------------------
def show_today_board(board, bonus, rack):
    print("Today's Board:")
    print_board(board, bonus)
    print()
    print(f"Rack: {' '.join(rack)}")
    print()


def summarize_fixed_words(board, bonus):
    """Small helper to list any fixed words already on the initial board."""
    mask = get_letter_mask(board)
    words = []
    # across
    for r in range(N):
        c = 0
        while c < N:
            if mask[r][c]:
                start = c
                while c < N and mask[r][c]:
                    c += 1
                if c - start >= 2:
                    letters = [board[r][i] for i in range(start, c)]
                    bonuses = bonus[r][start:c]
                    positions = [(r, cc) for cc in range(start, c)]
                    words.append((score_word(letters, bonuses), "".join(letters), positions))
            else:
                c += 1
    # down
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
                    words.append((score_word(letters, bonuses), "".join(letters), positions))
            else:
                r += 1
    if words:
        print(Fore.LIGHTBLUE_EX + "Fixed words on the board:")
        for sc, w, pos in words:
            print(Fore.LIGHTBLUE_EX + f"  {w} (score {sc}) at {pos}")
        print(Fore.RESET, end="")


# -------------------------------
# Solver
# -------------------------------
def run_solver():
    parser = argparse.ArgumentParser(description="ScrapleSolver")
    parser.add_argument("--beam-width", type=int, default=50, help="Beam width for the search (default: 50)")
    parser.add_argument("--num-games", type=int, default=50, help="Number of first-move candidates to simulate (default: 50)")
    parser.add_argument("--first-moves", type=int, default=None, help="Override number of opening moves to explore")
    parser.add_argument("--depth", type=int, default=20, help="Maximum number of moves to search (default: 20)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-cache", action="store_true", help="Disable board score caching")
    parser.add_argument("--log-puzzle", action="store_true", help="Save the day's puzzle and best result to a JSON log file")

    # Start word forcing (optional)
    parser.add_argument("--start-word", type=str, default=None, help="Specify a start word to force as the first move")
    parser.add_argument("--start-word-pos", type=str, default=None, help="row,col,dir (e.g. 1,0,A) for forced start word")

    # Beam heuristics
    parser.add_argument("--alpha-premium", type=float, default=0.5, help="Weight for premium coverage shaping")
    parser.add_argument("--beta-mobility", type=float, default=0.2, help="Weight for anchor/mobility shaping")
    parser.add_argument("--gamma-diversity", type=float, default=0.01, help="Penalty per repeated (r,c,dir) choice")

    # Transposition table
    parser.add_argument("--use-transpo", action="store_true", help="Enable transposition table")
    parser.add_argument("--transpo-cap", type=int, default=200000, help="Max transposition entries")

    # Generator mode
    parser.add_argument("--gen", type=str, choices=["anchor", "scan"], default="anchor",
                        help="Candidate generator: 'anchor' = DAWG+anchors (hybrid pad), 'scan' = legacy scanner")

    # Hybrid legacy pad knobs (sane defaults; small and capped in search.py)
    parser.add_argument("--legacy-pad-k", type=int, default=4, help="Extra legacy candidates to add each move (cap ~6)")
    parser.add_argument("--legacy-pad-ratio", type=float, default=0.2, help="Pad size as fraction of beam width")

    # Power user: try to beat current leaderboard high score board directly
    parser.add_argument("--improve-leaderboard", action="store_true",
                        help="If present, simulate starting from current leaderboard best board state")
    
    parser.add_argument("--empty-anchor-cap", type=int, default=250,
                    help="Max number of anchor emits on an empty board (default: 250)")

    args = parser.parse_args()

    empty_anchor_cap = args.empty_anchor_cap

    # Verbosity + caching flags
    utils.VERBOSE = args.verbose
    if args.no_cache:
        from score_cache import DISABLE_CACHE
        DISABLE_CACHE = True

    # --- FIX: ensure utils.start_time is initialized even if the module defines it as None ---
    if getattr(utils, "start_time", None) is None:
        utils.start_time = time.time()

    # Pull today's puzzle, dictionary, and leaderboard concurrently (unless loading a saved log)
    if args.start_word:
        args.start_word = args.start_word.strip().upper()

    # Possibly load from a previous log (for deterministic reruns)
    leaderboard_data = None
    board = None
    rack = None
    dict_text = ""
    words = []
    wordset = set()
    prefixset = set()

    if getattr(args, "load_log", None):
        try:
            with open(args.load_log, "r") as f:
                log_data = json.load(f)
            puzzle = log_data["puzzle"]
            board = [["" for _ in range(N)] for _ in range(N)]
            for bonus, pos in puzzle["bonusTilePositions"].items():
                if isinstance(pos, (list, tuple)):
                    r, c = pos
                else:
                    r, c = pos["row"], pos["col"]
                board[r][c] = MAPPING[bonus]
            rack = [t["letter"].upper() for t in puzzle["letters"]]
            leaderboard_data = None
            words, wordset, prefixset = load_dictionary()[:3]
        except Exception as e:
            log_with_time(f"Error loading log file: {e}", color=Fore.RED)
            return
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            fx_puzzle = executor.submit(fetch_board_and_rack)
            fx_dict = executor.submit(load_dictionary)
            fx_lb = executor.submit(lambda: requests.get(LEADERBOARD_URL, timeout=10))

            board, rack, puzzle_json = fx_puzzle.result()

            # dict
            dict_result = fx_dict.result()
            if isinstance(dict_result, tuple) and len(dict_result) == 4:
                words, wordset, prefixset, dict_text = dict_result
            elif isinstance(dict_result, tuple) and len(dict_result) == 3:
                words, wordset, prefixset = dict_result
            else:
                raise RuntimeError("Bad dictionary result")

            # leaderboard
            try:
                leaderboard_data = parse_leaderboard(fx_lb.result())
            except Exception:
                leaderboard_data = None

    # Build DAWG if available
    dawg = None
    if DAWG is not None:
        try:
            t_dawg = time.time()
            dawg = DAWG.build(words)
            t_dawg = (time.time() - t_dawg) * 1000
            log_with_time(f"DAWG built: {'yes' if dawg else 'no'} ({t_dawg:.0f} ms)")
        except Exception as e:
            log_with_time(f"DAWG build failed, falling back to scan: {e}", color=Fore.YELLOW)
            dawg = None
    else:
        log_with_time("DAWG module not available — using legacy scan", color=Fore.YELLOW)

    # Show today's board and rack
    original_bonus = [row[:] for row in board]
    show_today_board(board, original_bonus, rack)
    summarize_fixed_words(board, original_bonus)

    # Show current leaderboard best layout (if any)
    if leaderboard_data:
        try:
            leaderboard_scores = [entry["score"] for entry in leaderboard_data.get("scores", [])]
            if leaderboard_scores:
                high_score = max(leaderboard_scores)
                top_entries = [e for e in leaderboard_data.get("scores", []) if e["score"] == high_score]

                print(Fore.LIGHTYELLOW_EX + f"\nCurrent High Score Board Layout (Score: {high_score}):")
                any_printed = False
                for entry in top_entries:
                    gs = entry.get("gameState")
                    if not gs:
                        continue
                    try:
                        b_hs, bonus_hs = leaderboard_gamestate_to_board(gs)
                        print_board(b_hs, bonus_hs)
                        any_printed = True
                    except Exception as e:
                        print(Fore.LIGHTYELLOW_EX + f"    (Could not display board: {e})")
                if any_printed:
                    print(Fore.RESET, end="")
        except Exception:
            pass

    # Argument/plumbing setup
    beam_width = args.beam_width
    num_games = args.num_games
    # If first_moves not provided, default to num_games (not beam width).
    first_moves = args.first_moves if args.first_moves is not None else num_games
    max_moves = args.depth
    alpha_premium = args.alpha_premium
    beta_mobility = args.beta_mobility
    gamma_diversity = args.gamma_diversity
    use_transpo = args.use_transpo
    transpo_cap = args.transpo_cap

    use_anchor_gen = (args.gen == "anchor")
    legacy_pad_k = args.legacy_pad_k
    legacy_pad_ratio = args.legacy_pad_ratio

    # If a start word is forced, create that placement then continue beam from there
    best_total = float("-inf")
    best_results = []

    if args.start_word:
        # Determine placement: explicit pos or best legal placement
        placement = None
        rack_counter = Counter(rack)

        if args.start_word_pos:
            try:
                row_s, col_s, dir_s = args.start_word_pos.split(",")
                row = int(row_s)
                col = int(col_s)
                dir_enum = Direction.ACROSS if dir_s.strip().upper().startswith("A") else Direction.DOWN
                # Validate placement against current board
                cands = find_best(
                    board, rack_counter, [args.start_word], wordset,
                    prefixset, None, original_bonus, top_k=None, dawg=dawg
                )
                for p in cands:
                    if p[2] == dir_enum and p[3] == row and p[4] == col:
                        placement = p
                        break
                if not placement:
                    log_with_time(f"Cannot place start word '{args.start_word}' at {row},{col},{dir_s}.", color=Fore.RED)
                    return
            except Exception as e:
                log_with_time(f"Invalid --start-word-pos. Use row,col,dir (e.g. 1,0,A). Error: {e}", color=Fore.RED)
                return
        else:
            cands = find_best(
                board, rack_counter, [args.start_word], wordset,
                prefixset, None, original_bonus, top_k=None, dawg=dawg
            )
            if not cands:
                log_with_time(f"No valid placements for start word '{args.start_word}' on the board.", color=Fore.RED)
                return
            placement = max(cands, key=lambda x: x[0])

        log_with_time(
            f"Best placement for '{args.start_word}': score {placement[0]}, position {placement[3]},{placement[4]},{placement[2].value}",
            color=Fore.YELLOW,
        )

        # Run beam from that first move
        score, final_board, moves = beam_from_first(
            placement, board, rack_counter, words, wordset, original_bonus,
            beam_width=beam_width, max_moves=max_moves, prefixset=prefixset,
            alpha_premium=alpha_premium, beta_mobility=beta_mobility, gamma_diversity=gamma_diversity,
            use_transpo=use_transpo, transpo_cap=transpo_cap,
            dawg=dawg, use_anchor_gen=use_anchor_gen,
            legacy_pad_k=legacy_pad_k, legacy_pad_ratio=legacy_pad_ratio,
            empty_anchor_cap=empty_anchor_cap,
        )
        if final_board:
            best_total = score
            best_results = [(score, final_board, moves)]
    else:
        # Normal: parallel first-move exploration
        best_total, best_results = parallel_first_beam(
            board, rack, words, wordset, original_bonus,
            beam_width=beam_width, num_games=num_games, first_moves=first_moves, max_moves=max_moves,
            prefixset=prefixset, alpha_premium=alpha_premium, beta_mobility=beta_mobility, gamma_diversity=gamma_diversity,
            use_transpo=use_transpo, transpo_cap=transpo_cap, dawg=dawg,
            use_anchor_gen=use_anchor_gen, legacy_pad_k=legacy_pad_k, legacy_pad_ratio=legacy_pad_ratio,
            empty_anchor_cap=empty_anchor_cap, 
        )

    # Present results
    if not best_results:
        log_with_time("No valid full simulation found.")
        return

    log_with_time(f"Found {len(best_results)} highest scoring solution(s) with score {best_total}:", color=Fore.GREEN)
    for idx, (score, best_board, best_moves) in enumerate(best_results, 1):
        log_with_time(f"Solution {idx}:", color=Fore.GREEN)
        log_with_time("Move sequence:", color=Fore.GREEN)
        for sc, w, d, r0, c0 in best_moves:
            log_with_time(f"  {w} at {r0},{c0},{d.value} scoring {sc}", color=Fore.GREEN)
        log_with_time("Final simulated board:", color=Fore.GREEN)
        print()
        print_board(best_board, original_bonus)
        print(f"Final board score: {cached_board_score(board_to_tuple(best_board), board_to_tuple(original_bonus))}")
        print("-" * 40)

    # Compare vs leaderboard (if we fetched it)
    if leaderboard_data:
        print_leaderboard_summary(best_total, leaderboard_data)

    total_elapsed = time.time() - utils.start_time
    print(f"Total time: {int(total_elapsed // 60)}m {total_elapsed % 60:.1f}s")


if __name__ == "__main__":
    run_solver()