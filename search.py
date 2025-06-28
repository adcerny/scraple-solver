# --- search.py ---

from collections import Counter
import concurrent.futures
import time
from utils import log_with_time, vlog, N
from board import board_valid, place_word
from score_cache import cached_board_score, board_to_tuple

def can_play_word_on_board(word, r0, c0, d, board, rack):
    rack = rack.copy()
    for i, ch in enumerate(word):
        r = r0 + (i if d == 'V' else 0)
        c = c0 + (i if d == 'H' else 0)
        if not (len(board[r][c]) == 1):
            if rack[ch] > 0:
                rack[ch] -= 1
            else:
                return False, None
    return True, rack

def get_perpendicular_coords(temp, r, c, direction):
    coords = [(r, c)]
    if direction == 'H':
        i = r - 1
        while i >= 0 and len(temp[i][c]) == 1:
            coords.insert(0, (i, c)); i -= 1
        i = r + 1
        while i < N and len(temp[i][c]) == 1:
            coords.append((i, c)); i += 1
    else:
        j = c - 1
        while j >= 0 and len(temp[r][j]) == 1:
            coords.insert(0, (r, j)); j -= 1
        j = c + 1
        while j < N and len(temp[r][j]) == 1:
            coords.append((r, j)); j += 1
    return coords if len(coords) > 1 else []

def is_valid_placement(w, board, rack_count, wordset, r0, c0, d):
    needed = Counter()
    for i, ch in enumerate(w):
        r = r0 + (i if d == 'V' else 0)
        c = c0 + (i if d == 'H' else 0)
        if not (0 <= r < N and 0 <= c < N): return False
        cell = board[r][c]
        if len(cell) == 1:
            if cell != ch: return False
        else:
            needed[ch] += 1
    if any(rack_count[ch] < needed[ch] for ch in needed): return False
    if sum(needed.values()) == 0: return False
    temp = [row[:] for row in board]
    for i, ch in enumerate(w):
        r = r0 + (i if d == 'V' else 0); c = c0 + (i if d == 'H' else 0)
        if not (len(temp[r][c]) == 1): temp[r][c] = ch
    for i, ch in enumerate(w):
        r = r0 + (i if d == 'V' else 0); c = c0 + (i if d == 'H' else 0)
        coords = get_perpendicular_coords(temp, r, c, d)
        if coords:
            word = ''.join(temp[rr][cc] for rr, cc in coords)
            if word not in wordset: return False
    return True

def prune_words(words, rack_count, board):
    t0 = time.time()
    board_letters = Counter(cell for row in board for cell in row if len(cell) == 1)
    rack_plus_board = rack_count + board_letters
    pruned = []
    for w in words:
        wc = Counter(w)
        if all(rack_plus_board[ch] >= wc[ch] for ch in wc):
            pruned.append(w)
    vlog(f"prune_words: reduced from {len(words)} to {len(pruned)}", t0)
    return pruned

def validate_new_words(board, wordset, w, r0, c0, d):
    # Check the main word
    main_word = []
    if d == 'H':
        c_start = c0
        while c_start > 0 and len(board[r0][c_start-1]) == 1:
            c_start -= 1
        c_end = c0 + len(w)
        while c_end < N and len(board[r0][c_end]) == 1:
            c_end += 1
        main_word = ''.join(board[r0][c] for c in range(c_start, c_end))
    else:
        r_start = r0
        while r_start > 0 and len(board[r_start-1][c0]) == 1:
            r_start -= 1
        r_end = r0 + len(w)
        while r_end < N and len(board[r_end][c0]) == 1:
            r_end += 1
        main_word = ''.join(board[r][c0] for r in range(r_start, r_end))
    if len(main_word) > 1 and main_word not in wordset:
        return False
    # Check all perpendicular words formed by new tiles
    for i, ch in enumerate(w):
        r = r0 + (i if d == 'V' else 0)
        c = c0 + (i if d == 'H' else 0)
        if len(board[r][c]) != 1:  # Only check for new tiles placed
            continue
        # Build perpendicular word
        if d == 'H':
            r_start = r
            while r_start > 0 and len(board[r_start-1][c]) == 1:
                r_start -= 1
            r_end = r + 1
            while r_end < N and len(board[r_end][c]) == 1:
                r_end += 1
            if r_end - r_start > 1:
                perp_word = ''.join(board[rr][c] for rr in range(r_start, r_end))
                if perp_word not in wordset:
                    return False
        else:
            c_start = c
            while c_start > 0 and len(board[r][c_start-1]) == 1:
                c_start -= 1
            c_end = c + 1
            while c_end < N and len(board[r][c_end]) == 1:
                c_end += 1
            if c_end - c_start > 1:
                perp_word = ''.join(board[r][cc] for cc in range(c_start, c_end))
                if perp_word not in wordset:
                    return False
    return True

def find_best(board, rack_count, words, wordset, touch=None, original_bonus=None, top_k=10):
    t0 = time.time()
    base_score = cached_board_score(board_to_tuple(board), board_to_tuple(original_bonus))
    checked = 0
    candidates = []
    for w in words:
        L = len(w)
        for r in range(N):
            for c in range(N-L+1):
                if not is_valid_placement(w, board, rack_count, wordset, r, c, 'H'): continue
                coords = [(r, c+i) for i in range(L)]
                if touch and not any((nr,nc) in touch or any(abs(nr-tr)+abs(nc-tc)==1 for tr,tc in touch) for nr,nc in coords): continue
                board_copy = [row[:] for row in board]
                rack_copy = rack_count.copy()
                can_play, _ = can_play_word_on_board(w, r, c, 'H', board_copy, rack_copy)
                if not can_play: continue
                place_word(board_copy, w, r, c, 'H')
                if not validate_new_words(board_copy, wordset, w, r, c, 'H'): continue
                move_score = cached_board_score(board_to_tuple(board_copy), board_to_tuple(original_bonus))
                bonus_count = sum(1 for i in range(L) if board[r][c+i] in {'DL','TL','DW','TW'})
                candidates.append((move_score, bonus_count, L, w, 'H', r, c))
                checked += 1
        for r in range(N-L+1):
            for c in range(N):
                if not is_valid_placement(w, board, rack_count, wordset, r, c, 'V'): continue
                coords = [(r+i, c) for i in range(L)]
                if touch and not any((nr,nc) in touch or any(abs(nr-tr)+abs(nc-tc)==1 for tr,tc in touch) for nr,nc in coords): continue
                board_copy = [row[:] for row in board]
                rack_copy = rack_count.copy()
                can_play, _ = can_play_word_on_board(w, r, c, 'V', board_copy, rack_copy)
                if not can_play: continue
                place_word(board_copy, w, r, c, 'V')
                if not validate_new_words(board_copy, wordset, w, r, c, 'V'): continue
                move_score = cached_board_score(board_to_tuple(board_copy), board_to_tuple(original_bonus))
                bonus_count = sum(1 for i in range(L) if board[r+i][c] in {'DL','TL','DW','TW'})
                candidates.append((move_score, bonus_count, L, w, 'V', r, c))
                checked += 1
    # Sort by move_score, then bonus_count, then word length (descending)
    candidates.sort(reverse=True)
    vlog(f"find_best checked {checked} placements for {len(words)} words, returning top {top_k}", t0)
    # Early pruning: only return the top_k best candidates
    if candidates:
        best = candidates[0]
        return (best[0], best[3], best[4], best[5], best[6])
    return (float('-inf'), None, None, None, None)

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
                if not validate_new_words(b2, wordset, w, r0, c0, d):
                    temp_words = [x for x in temp_words if x != w]
                    continue
                pl2 = {(r, c) for r in range(N) for c in range(N) if len(b2[r][c]) == 1}
                next_words = [x for x in temp_words if x != w]
                next_state.append((cached_board_score(board_to_tuple(b2), board_to_tuple(original_bonus)), b2, rack_after, pl2, moves + [(sc, w, d, r0, c0)], next_words))
                temp_words = next_words
        vlog(f"full_beam_search move {move_num}: {len(state)} states expanded to {len(next_state)}", t0)
        state = sorted(next_state, key=lambda x: x[0], reverse=True)[:beam_width]
        if state and state[0][0] > best_score:
            best_score = state[0][0]
            best_board = state[0][1]
            best_moves = state[0][4]
        move_num += 1
    return best_score, best_board, best_moves

def beam_from_first(play, board, rack_count, words, wordset, original_bonus, beam_width, max_moves=20):
    play_word = play[1]
    words_for_sim = [w for w in words if w != play_word]
    board_copy = [row[:] for row in board]
    rack_count_copy = rack_count.copy()
    can_play, rack_after_first = can_play_word_on_board(play_word, play[3], play[4], play[2], board_copy, rack_count_copy)
    if not can_play:
        return (float('-inf'), None, None)
    place_word(board_copy, play_word, play[3], play[4], play[2])
    placed_copy = {(r, c) for r in range(N) for c in range(N) if len(board_copy[r][c]) == 1}
    score, final_board, moves = full_beam_search(
        board_copy, rack_after_first, words_for_sim, wordset, placed_copy, original_bonus, beam_width=beam_width, max_moves=max_moves
    )
    return (score, final_board, [(play[0], play_word, play[2], play[3], play[4])] + (moves if moves else []))

def parallel_first_beam(board, rack, words, wordset, original_bonus, beam_width=5, max_moves=20):
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
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_start = {
            executor.submit(
                beam_from_first, play, board, rack_count, pruned_words, wordset, original_bonus, beam_width, max_moves
            ): time.time()
            for play in first_choices
        }
        for i, future in enumerate(concurrent.futures.as_completed(future_to_start)):
            start = future_to_start[future]
            score, board_result, moves = future.result()
            elapsed = time.time() - start
            _, word, direction, row, col = first_choices[i]
            color = "\033[92m" if score == max(r[0] for r in results + [(score, None, None)]) else "\033[94m"
            reset = "\033[0m"
            log_with_time(f"{color}Move {i+1}/{len(first_choices)}: {word} at ({row},{col}) {direction} → final score: {score} (duration: {elapsed:.3f}s){reset}")
            vlog(f"beam_from_first {i+1}", start)
            if moves is not None:
                results.append((score, board_result, moves))

    if not results:
        return 0, []

    max_score = max(r[0] for r in results)
    best_results = [r for r in results if r[0] == max_score]
    return max_score, best_results