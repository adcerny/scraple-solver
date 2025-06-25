# scraple-solver.py

import requests
from collections import Counter
from colorama import init, Fore, Style
import time
import argparse
import concurrent.futures

# Initialise ANSI colours on Windows
init(autoreset=True)

API_URL  = 'https://scraple.io/api/daily-puzzle'
DICT_URL = 'https://scraple.io/dictionary.txt'
N        = 5  # board dimension

# Bonus‐square codes
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

def vlog(msg, start_time=None):
    """Verbose log with optional duration."""
    if VERBOSE:
        if start_time is not None:
            elapsed = time.time() - start_time
            log_with_time(f"{msg} (took {elapsed:.3f}s)")
        else:
            log_with_time(msg)

def fetch_board_and_rack():
    resp = requests.get(API_URL); resp.raise_for_status()
    data = resp.json()
    board = [['' for _ in range(N)] for _ in range(N)]
    for bonus, (r, c) in data['bonusTilePositions'].items():
        board[r][c] = MAPPING[bonus]
    rack = [t['letter'].upper() for t in data['letters']]
    return board, rack

def load_dictionary():
    t0 = time.time()
    log_with_time("⟳ Downloading dictionary…")
    resp = requests.get(DICT_URL); resp.raise_for_status()
    words = [
        w.strip().upper()
        for w in resp.text.splitlines()
        if w.strip().isalpha() and 2 <= len(w.strip()) <= N
    ]
    wordset = set(words)
    vlog(f"Dictionary loaded and filtered ({len(words)} words)", t0)
    log_with_time(f"✅ {len(words)} words")
    return words, wordset

def is_letter(cell):
    return len(cell) == 1

def get_perpendicular_coords(temp, r, c, direction):
    coords = [(r, c)]
    if direction == 'H':
        i = r - 1
        while i >= 0 and is_letter(temp[i][c]):
            coords.insert(0, (i, c)); i -= 1
        i = r + 1
        while i < N and is_letter(temp[i][c]):
            coords.append((i, c)); i += 1
    else:
        j = c - 1
        while j >= 0 and is_letter(temp[r][j]):
            coords.insert(0, (r, j)); j -= 1
        j = c + 1
        while j < N and is_letter(temp[r][j]):
            coords.append((r, j)); j += 1
    return coords if len(coords) > 1 else []

def is_valid_placement(w, board, rack_count, wordset, r0, c0, d):
    needed = Counter()
    for i, ch in enumerate(w):
        r = r0 + (i if d == 'V' else 0)
        c = c0 + (i if d == 'H' else 0)
        if not (0 <= r < N and 0 <= c < N): return False
        cell = board[r][c]
        if is_letter(cell):
            if cell != ch: return False
        else:
            needed[ch] += 1
    if any(rack_count[ch] < needed[ch] for ch in needed): return False
    if sum(needed.values()) == 0: return False
    temp = [row[:] for row in board]
    for i, ch in enumerate(w):
        r = r0 + (i if d == 'V' else 0); c = c0 + (i if d == 'H' else 0)
        if not is_letter(temp[r][c]): temp[r][c] = ch
    for i, ch in enumerate(w):
        r = r0 + (i if d == 'V' else 0); c = c0 + (i if d == 'H' else 0)
        coords = get_perpendicular_coords(temp, r, c, d)
        if coords:
            word = ''.join(temp[rr][cc] for rr, cc in coords)
            if word not in wordset: return False
    return True

def place_word(board, w, r0, c0, d):
    for i, ch in enumerate(w): board[r0 + (i if d=='V' else 0)][c0 + (i if d=='H' else 0)] = ch

def print_board(board):
    for row in board:
        line = []
        for cell in row:
            if cell == 'DL': line.append(Fore.BLUE + 'DL' + Style.RESET_ALL)
            elif cell == 'TL': line.append(Fore.CYAN + 'TL' + Style.RESET_ALL)
            elif cell == 'DW': line.append(Fore.MAGENTA + 'DW' + Style.RESET_ALL)
            elif cell == 'TW': line.append(Fore.RED + 'TW' + Style.RESET_ALL)
            elif is_letter(cell): line.append(Fore.GREEN + f' {cell}' + Style.RESET_ALL)
            else: line.append('..')
        print(' '.join(line))
    print()

def can_play_word_on_board(word, r0, c0, d, board, rack):
    rack = rack.copy()
    for i, ch in enumerate(word):
        r = r0 + (i if d == 'V' else 0)
        c = c0 + (i if d == 'H' else 0)
        if not is_letter(board[r][c]):
            if rack[ch] > 0:
                rack[ch] -= 1
            else:
                return False, None
    return True, rack

def compute_board_score(board, original_bonus):
    N = len(board)
    total = 0
    # Score all horizontal words
    for r in range(N):
        c = 0
        while c < N:
            if is_letter(board[r][c]):
                start = c
                while c < N and is_letter(board[r][c]): c += 1
                length = c - start
                if length >= 2:
                    sc, wm = 0, 1
                    for i in range(start, c):
                        ch = board[r][i]
                        cell = original_bonus[r][i]
                        val = LETTER_SCORES[ch]
                        if cell == 'DL': val *= 2
                        elif cell == 'TL': val *= 3
                        if cell == 'DW': wm *= 2
                        elif cell == 'TW': wm *= 3
                        sc += val
                    total += sc * wm
            else:
                c += 1
    # Score all vertical words
    for c in range(N):
        r = 0
        while r < N:
            if is_letter(board[r][c]):
                start = r
                while r < N and is_letter(board[r][c]): r += 1
                length = r - start
                if length >= 2:
                    sc, wm = 0, 1
                    for i in range(start, r):
                        ch = board[i][c]
                        cell = original_bonus[i][c]
                        val = LETTER_SCORES[ch]
                        if cell == 'DL': val *= 2
                        elif cell == 'TL': val *= 3
                        if cell == 'DW': wm *= 2
                        elif cell == 'TW': wm *= 3
                        sc += val
                    total += sc * wm
            else:
                r += 1
    return total

def board_valid(board, wordset):
    for r in range(N):
        c = 0
        while c < N:
            if is_letter(board[r][c]):
                start = c
                while c < N and is_letter(board[r][c]): c += 1
                length = c - start
                if length >= 2:
                    w = ''.join(board[r][start:start+length])
                    if w not in wordset: return False
            else:
                c += 1
    for c in range(N):
        r = 0
        while r < N:
            if is_letter(board[r][c]):
                start = r
                while r < N and is_letter(board[r][c]): r += 1
                length = r - start
                if length >= 2:
                    w = ''.join(board[i][c] for i in range(start, r))
                    if w not in wordset: return False
            else:
                r += 1
    return True

def prune_words(words, rack_count, board):
    t0 = time.time()
    board_letters = Counter(cell for row in board for cell in row if is_letter(cell))
    rack_plus_board = rack_count + board_letters
    pruned = []
    for w in words:
        wc = Counter(w)
        if all(rack_plus_board[ch] >= wc[ch] for ch in wc):
            pruned.append(w)
    vlog(f"prune_words: reduced from {len(words)} to {len(pruned)}", t0)
    return pruned

def find_best(board, rack_count, words, wordset, touch=None, original_bonus=None):
    t0 = time.time()
    best = (float('-inf'), None, None, None, None)
    base_score = compute_board_score(board, original_bonus)
    checked = 0
    for w in words:
        L = len(w)
        for r in range(N):
            for c in range(N-L+1):
                if not is_valid_placement(w, board, rack_count, wordset, r, c, 'H'): continue
                coords = [(r, c+i) for i in range(L)]
                if touch and not any((nr,nc) in touch or any(abs(nr-tr)+abs(nc-tc)==1 for tr,tc in touch) for nr,nc in coords): continue
                board_copy = [row[:] for row in board]
                rack_copy = rack_count.copy()
                can_play, rack_after = can_play_word_on_board(w, r, c, 'H', board_copy, rack_copy)
                if not can_play: continue
                place_word(board_copy, w, r, c, 'H')
                if not board_valid(board_copy, wordset): continue
                move_score = compute_board_score(board_copy, original_bonus) - base_score
                if move_score > best[0]: best = (move_score, w, 'H', r, c)
                checked += 1
        for r in range(N-L+1):
            for c in range(N):
                if not is_valid_placement(w, board, rack_count, wordset, r, c, 'V'): continue
                coords = [(r+i, c) for i in range(L)]
                if touch and not any((nr,nc) in touch or any(abs(nr-tr)+abs(nc-tc)==1 for tr,tc in touch) for nr,nc in coords): continue
                board_copy = [row[:] for row in board]
                rack_copy = rack_count.copy()
                can_play, rack_after = can_play_word_on_board(w, r, c, 'V', board_copy, rack_copy)
                if not can_play: continue
                place_word(board_copy, w, r, c, 'V')
                if not board_valid(board_copy, wordset): continue
                move_score = compute_board_score(board_copy, original_bonus) - base_score
                if move_score > best[0]: best = (move_score, w, 'V', r, c)
                checked += 1
    vlog(f"find_best checked {checked} placements for {len(words)} words", t0)
    return best

def full_beam_search(board, rack_count, words, wordset, placed, original_bonus, beam_width=5, max_moves=20):
    state = [(0, board, rack_count, set(placed), [], words)]
    best_score = 0
    best_board = None
    best_moves = None
    move_num = 1

    while state and move_num <= max_moves:
        t0 = time.time()
        next_state = []
        for score, b, rc, pl, moves, rem_words in state:
            touch = None if not moves else pl
            pruned_words = prune_words(rem_words, rc, b)
            temp_words = pruned_words.copy()
            for _ in range(beam_width):
                sc, w, d, r0, c0 = find_best(b, rc, temp_words, wordset, touch, original_bonus)
                if not w:
                    break
                can_play, rack_after = can_play_word_on_board(w, r0, c0, d, b, rc)
                if not can_play:
                    temp_words = [x for x in temp_words if x != w]
                    continue
                b2 = [row[:] for row in b]
                place_word(b2, w, r0, c0, d)
                if not board_valid(b2, wordset):
                    temp_words = [x for x in temp_words if x != w]
                    continue
                pl2 = {(r, c) for r in range(N) for c in range(N) if is_letter(b2[r][c])}
                next_words = [x for x in temp_words if x != w]
                next_state.append((compute_board_score(b2, original_bonus), b2, rack_after, pl2, moves + [(sc, w, d, r0, c0)], next_words))
                temp_words = next_words
        vlog(f"full_beam_search move {move_num}: {len(state)} states expanded to {len(next_state)}", t0)
        state = sorted(next_state, key=lambda x: x[0], reverse=True)[:beam_width]
        if state:
            if state[0][0] > best_score:
                best_score = state[0][0]
                best_board = state[0][1]
                best_moves = state[0][4]
        move_num += 1
    return best_score, best_board, best_moves

def parallel_first_beam(board, rack, words, wordset, original_bonus, beam_width=5):
    rack_count = Counter(rack)
    placed = set()
    t0 = time.time()
    pruned_words = prune_words(words, rack_count, board)
    log_with_time(f"Pruned word list: {len(pruned_words)} words")
    vlog("Initial prune_words", t0)
    temp_words = pruned_words.copy()
    first_choices = []
    for _ in range(beam_width):
        t1 = time.time()
        sc, w, d, r0, c0 = find_best(board, rack_count, temp_words, wordset, None, original_bonus)
        vlog(f"find_best for first move {_+1}", t1)
        if not w:
            break
        first_choices.append((sc, w, d, r0, c0))
        temp_words = [x for x in temp_words if x != w]

    results = []

    # --- Parallelize the first move simulations ---
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Store (future, start_time) pairs
        future_to_start = {
            executor.submit(
                beam_from_first, play, board, rack_count, pruned_words, wordset, original_bonus, beam_width
            ): time.time()
            for play in first_choices
        }
        for i, future in enumerate(concurrent.futures.as_completed(future_to_start)):
            start = future_to_start[future]
            score, board_result, moves = future.result()
            elapsed = time.time() - start
            log_with_time(f"First move {i+1}/{len(first_choices)} done, score: {score} (took {elapsed:.3f}s)")
            vlog(f"beam_from_first {i+1}", start)
            if moves is not None:
                results.append((score, board_result, moves))

    if not results:
        return 0, [], []

    # Find all with the highest score
    max_score = max(r[0] for r in results)
    best_results = [r for r in results if r[0] == max_score]
    return max_score, best_results

def beam_from_first(play, board, rack_count, words, wordset, original_bonus, beam_width):
    play_word = play[1]
    words_for_sim = [w for w in words if w != play_word]
    board_copy = [row[:] for row in board]
    rack_count_copy = rack_count.copy()
    can_play, rack_after_first = can_play_word_on_board(play_word, play[3], play[4], play[2], board_copy, rack_count_copy)
    if not can_play:
        return (float('-inf'), None, None)
    place_word(board_copy, play_word, play[3], play[4], play[2])
    placed_copy = {(r, c) for r in range(N) for c in range(N) if is_letter(board_copy[r][c])}
    score, final_board, moves = full_beam_search(
        board_copy, rack_after_first, words_for_sim, wordset, placed_copy, original_bonus, beam_width=beam_width
    )
    return (score, final_board, [(play[0], play_word, play[2], play[3], play[4])] + (moves if moves else []))

def main():
    global start_time, VERBOSE
    start_time = time.time()

    parser = argparse.ArgumentParser(description="ScrapleSolver")
    parser.add_argument('--beam-width', type=int, default=5, help='Beam width for the search (default: 5)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    VERBOSE = args.verbose

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

    total_elapsed = time.time() - start_time
    print(f"Total time: {int(total_elapsed // 60)}m {total_elapsed % 60:.1f}s")

if __name__ == '__main__':
    main()