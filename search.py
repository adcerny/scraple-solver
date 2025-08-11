from collections import Counter, defaultdict
import concurrent.futures
import time
import hashlib
from colorama import Fore, Style
from utils import log_with_time, vlog, N, PRINT_LOCK, VERBOSE, Direction
from board import board_valid, place_word, print_board
from score_cache import cached_board_score, board_to_tuple, board_hash

# ---- Default heuristic weights (can be overridden via params from solver) ----
ALPHA_PREMIUM_DEFAULT = 0.5   # weight for premium_coverage
BETA_MOBILITY_DEFAULT = 0.2   # weight for future_mobility (anchors)
GAMMA_DIVERSITY_DEFAULT = 0.01  # penalty per repeat (row,col,dir) at a ply


def _token64(word: str) -> int:
    """Deterministic 64-bit token for a word (for Zobrist-style hashing)."""
    h = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little")


def can_play_word_on_board(word, r0, c0, d, board, rack, wordset=None, prefixset=None):
    """Fast precheck using rack counts, prefix pruning, and early perpendicular checks.
    Does not modify the board. Returns (ok, new_rack_counter_or_None).
    """
    rack = rack.copy()

    # Existing prefix before the starting cell (helps prefix pruning)
    pr, pc = r0, c0
    if d == Direction.ACROSS:
        while pc > 0 and len(board[r0][pc - 1]) == 1:
            pc -= 1
        existing_prefix = "".join(board[r0][c] for c in range(pc, c0))
    else:
        while pr > 0 and len(board[pr - 1][c0]) == 1:
            pr -= 1
        existing_prefix = "".join(board[r][c0] for r in range(pr, r0))

    def check_cross(r, c, ch):
        # Build perpendicular word around (r, c) if any letters are present
        if d == Direction.ACROSS:
            rr1 = r - 1
            while rr1 >= 0 and len(board[rr1][c]) == 1:
                rr1 -= 1
            rr2 = r + 1
            while rr2 < N and len(board[rr2][c]) == 1:
                rr2 += 1
            if rr2 - (rr1 + 1) >= 1:
                parts = [board[rr][c] for rr in range(rr1 + 1, r)]
                parts.append(ch)
                parts += [board[rr][c] for rr in range(r + 1, rr2)]
                w = "".join(parts)
                if len(w) > 1:
                    return (wordset is None) or (w in wordset)
        else:
            cc1 = c - 1
            while cc1 >= 0 and len(board[r][cc1]) == 1:
                cc1 -= 1
            cc2 = c + 1
            while cc2 < N and len(board[r][cc2]) == 1:
                cc2 += 1
            if cc2 - (cc1 + 1) >= 1:
                parts = [board[r][cc] for cc in range(cc1 + 1, c)]
                parts.append(ch)
                parts += [board[r][cc] for cc in range(c + 1, cc2)]
                w = "".join(parts)
                if len(w) > 1:
                    return (wordset is None) or (w in wordset)
        return True

    for i, ch in enumerate(word):
        r = r0 + (i if d == Direction.DOWN else 0)
        c = c0 + (i if d == Direction.ACROSS else 0)
        cell = board[r][c]

        if len(cell) == 1:
            if cell != ch:
                return False, None
        else:
            if rack.get(ch, 0) <= 0:
                return False, None
            if not check_cross(r, c, ch):
                return False, None
            rack[ch] -= 1

        # Main word prefix pruning (allow full words too)
        if prefixset is not None:
            prefix = existing_prefix + word[: i + 1]
            if len(prefix) >= 2 and (prefix not in prefixset) and (wordset is None or prefix not in wordset):
                return False, None

    return True, rack


def is_valid_placement(w, board, rack_count, wordset, r0, c0, d):
    """Check if ``w`` can be legally placed on ``board`` starting at ``r0,c0``."""
    placed = []  # positions of newly placed tiles
    needed = {}  # tile counts required from rack

    for i, ch in enumerate(w):
        r = r0 + (i if d == Direction.DOWN else 0)
        c = c0 + (i if d == Direction.ACROSS else 0)
        if r < 0 or r >= N or c < 0 or c >= N:
            return False
        cell = board[r][c]
        if len(cell) == 1:
            if cell != ch:
                return False
        else:
            cnt = needed.get(ch, 0) + 1
            if rack_count[ch] < cnt:
                return False
            needed[ch] = cnt
            placed.append((r, c, ch))

    if not placed:  # must place at least one tile
        return False

    # Validate perpendicular words created by the new tiles
    for r, c, ch in placed:
        orig = board[r][c]
        board[r][c] = ch
        if d == Direction.ACROSS:
            i = r
            while i > 0 and len(board[i - 1][c]) == 1:
                i -= 1
            word_chars = []
            while i < N and len(board[i][c]) == 1:
                word_chars.append(board[i][c])
                i += 1
        else:
            j = c
            while j > 0 and len(board[r][j - 1]) == 1:
                j -= 1
            word_chars = []
            while j < N and len(board[r][j]) == 1:
                word_chars.append(board[r][j])
                j += 1
        if len(word_chars) > 1 and "".join(word_chars) not in wordset:
            board[r][c] = orig
            return False
        board[r][c] = orig

    return True


def prune_words(words, rack_count, board):
    """Cheap prefilter: ensure rack + present letters can cover word multiset."""
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
    if d == Direction.ACROSS:
        c_start = c0
        while c_start > 0 and len(board[r0][c_start - 1]) == 1:
            c_start -= 1
        c_end = c0 + len(w)
        while c_end < N and len(board[r0][c_end]) == 1:
            c_end += 1
        main_word = "".join(board[r0][c] for c in range(c_start, c_end))
    else:
        r_start = r0
        while r_start > 0 and len(board[r_start - 1][c0]) == 1:
            r_start -= 1
        r_end = r0 + len(w)
        while r_end < N and len(board[r_end][c0]) == 1:
            r_end += 1
        main_word = "".join(board[r][c0] for r in range(r_start, r_end))
    if len(main_word) > 1 and main_word not in wordset:
        return False

    # Check all perpendicular words formed by new tiles
    for i, _ in enumerate(w):
        r = r0 + (i if d == Direction.DOWN else 0)
        c = c0 + (i if d == Direction.ACROSS else 0)
        if d == Direction.ACROSS:
            r_start = r
            while r_start > 0 and len(board[r_start - 1][c]) == 1:
                r_start -= 1
            r_end = r + 1
            while r_end < N and len(board[r_end][c]) == 1:
                r_end += 1
            if r_end - r_start > 1:
                perp_word = "".join(board[rr][c] for rr in range(r_start, r_end))
                if perp_word not in wordset:
                    return False
        else:
            c_start = c
            while c_start > 0 and len(board[r][c_start - 1]) == 1:
                c_start -= 1
            c_end = c + 1
            while c_end < N and len(board[r][c_end]) == 1:
                c_end += 1
            if c_end - c_start > 1:
                perp_word = "".join(board[r][cc] for cc in range(c_start, c_end))
                if perp_word not in wordset:
                    return False
    return True


def find_best(board, rack_count, words, wordset, prefixset=None, touch=None, original_bonus=None, top_k=10):
    """Return the best placements across all words.
    Returns a list of (score, word, direction, row, col).
    """
    t0 = time.time()
    checked = 0
    candidates = []

    for w in words:
        L = len(w)

        # ACROSS
        for r in range(N):
            for c in range(N - L + 1):
                if not is_valid_placement(w, board, rack_count, wordset, r, c, Direction.ACROSS):
                    continue
                if touch:
                    coords = [(r, c + i) for i in range(L)]
                    if not any(
                        (nr, nc) in touch
                        or any(abs(nr - tr) + abs(nc - tc) == 1 for (tr, tc) in touch)
                        for nr, nc in coords
                    ):
                        continue
                board_copy = [row[:] for row in board]
                rack_copy = rack_count.copy()
                can_play, _ = can_play_word_on_board(
                    w, r, c, Direction.ACROSS, board_copy, rack_copy, wordset, prefixset
                )
                if not can_play:
                    continue
                place_word(board_copy, w, r, c, Direction.ACROSS)
                if not validate_new_words(board_copy, wordset, w, r, c, Direction.ACROSS):
                    continue
                move_score = cached_board_score(board_to_tuple(board_copy), board_to_tuple(original_bonus))
                bonus_count = sum(1 for i in range(L) if board[r][c + i] in {"DL", "TL", "DW", "TW"})
                candidates.append((move_score, bonus_count, L, w, Direction.ACROSS, r, c))
                checked += 1

        # DOWN
        for r in range(N - L + 1):
            for c in range(N):
                if not is_valid_placement(w, board, rack_count, wordset, r, c, Direction.DOWN):
                    continue
                if touch:
                    coords = [(r + i, c) for i in range(L)]
                    if not any(
                        (nr, nc) in touch
                        or any(abs(nr - tr) + abs(nc - tc) == 1 for (tr, tc) in touch)
                        for nr, nc in coords
                    ):
                        continue
                board_copy = [row[:] for row in board]
                rack_copy = rack_count.copy()
                can_play, _ = can_play_word_on_board(
                    w, r, c, Direction.DOWN, board_copy, rack_copy, wordset, prefixset
                )
                if not can_play:
                    continue
                place_word(board_copy, w, r, c, Direction.DOWN)
                if not validate_new_words(board_copy, wordset, w, r, c, Direction.DOWN):
                    continue
                move_score = cached_board_score(board_to_tuple(board_copy), board_to_tuple(original_bonus))
                bonus_count = sum(1 for i in range(L) if board[r + i][c] in {"DL", "TL", "DW", "TW"})
                candidates.append((move_score, bonus_count, L, w, Direction.DOWN, r, c))
                checked += 1

    candidates.sort(reverse=True)  # by move_score, then bonus_count, then length (due to tuple order)
    vlog(
        f"find_best checked {checked} placements for {len(words)} words, returning top {top_k if top_k is not None else 'all'}",
        t0,
    )
    if not candidates:
        return []
    selected = candidates if top_k is None else candidates[:top_k]
    return [(sc, w, d, r, c) for sc, _, _, w, d, r, c in selected]


# ---- Mobility & premium coverage utilities ----
def count_future_mobility(board):
    """Number of empty cells orthogonally adjacent to at least one letter."""
    anchors = 0
    for r in range(N):
        for c in range(N):
            if len(board[r][c]) == 1:
                continue
            if (
                (r > 0 and len(board[r - 1][c]) == 1)
                or (r + 1 < N and len(board[r + 1][c]) == 1)
                or (c > 0 and len(board[r][c - 1]) == 1)
                or (c + 1 < N and len(board[r][c + 1]) == 1)
            ):
                anchors += 1
    return anchors


def premium_coverage_from_move(board_before, board_after, original_bonus, w, r0, c0, d):
    """Count how many premium cells (DL/TL/DW/TW) were newly covered by *new* tiles of this move."""
    covered = 0
    L = len(w)
    for i in range(L):
        r = r0 + (i if d == Direction.DOWN else 0)
        c = c0 + (i if d == Direction.ACROSS else 0)
        was_letter = len(board_before[r][c]) == 1
        is_letter = len(board_after[r][c]) == 1
        if not was_letter and is_letter and original_bonus[r][c] in {"DL", "TL", "DW", "TW"}:
            covered += 1
    return covered


def _rack_key(rack_counter):
    """Stable, compact key for a rack Counter."""
    # Sort by letter for canonical order
    return tuple(sorted((ch, n) for ch, n in rack_counter.items() if n > 0))


def full_beam_search(
    board,
    rack_count,
    words,
    wordset,
    prefixset,
    placed,
    original_bonus,
    beam_width=5,
    max_moves=20,
    alpha_premium=ALPHA_PREMIUM_DEFAULT,
    beta_mobility=BETA_MOBILITY_DEFAULT,
    gamma_diversity=GAMMA_DIVERSITY_DEFAULT,
    use_transpo=False,
    transpo_cap=200000,
    *,
    _word_tokens=None,
    _rem_zobrist=None,
):
    """Beam search with multi-objective priority, diversity, and optional transposition pruning.
       Now transposition key includes remaining-words Zobrist hash to avoid over-pruning."""
    # Prepare Zobrist bookkeeping if using transpo
    if use_transpo:
        if _word_tokens is None:
            # Build tokens once for all distinct words we'll see in this branch
            _word_tokens = {w: _token64(w) for w in set(words)}
        if _rem_zobrist is None:
            rz = 0
            # XOR tokens for each element of list (handles duplicates correctly)
            for w in words:
                rz ^= _word_tokens.get(w) or _token64(w)
            _rem_zobrist = rz

    # state entries: (priority_score, current_score, board, rack_count, placed_set, moves_list, remaining_words, rem_zobrist)
    init_score = cached_board_score(board_to_tuple(board), board_to_tuple(original_bonus)) if words else 0
    state = [(init_score, init_score, board, rack_count, set(placed), [], words, _rem_zobrist)]
    best_score = 0
    best_board = None
    best_moves = None
    move_num = 1

    # transposition table: (board_hash, rack_key, depth, rem_zobrist) -> best_score
    transpo = {} if use_transpo else None

    if not words:
        if board_valid(board, wordset):
            best_score = init_score
            return best_score, board, []

    while state and move_num <= max_moves:
        t0 = time.time()
        next_state = []
        # diversity tracking for this ply
        repeat_counts = defaultdict(int)

        for _, cur_score, b, rc, pl, moves, rem_words, rem_rz in state:
            touch = None if not moves else pl
            pruned_words = prune_words(rem_words, rc, b)
            candidates = find_best(
                b,
                rc,
                pruned_words,
                wordset,
                prefixset=prefixset,
                touch=touch,
                original_bonus=original_bonus,
                top_k=beam_width,
            )
            for sc, w, d, r0, c0 in candidates:
                can_play, rack_after = can_play_word_on_board(w, r0, c0, d, b, rc, wordset, prefixset)
                if not can_play:
                    continue
                b2 = [row[:] for row in b]
                place_word(b2, w, r0, c0, d)
                if not validate_new_words(b2, wordset, w, r0, c0, d):
                    continue

                # Optional transposition prune
                if use_transpo:
                    h = board_hash(board_to_tuple(b2), board_to_tuple(original_bonus))
                    rk = _rack_key(rack_after)
                    # compute next remaining-words zobrist (remove all occurrences of w)
                    count_removed = rem_words.count(w)
                    next_rz = rem_rz
                    if count_removed % 2 == 1:
                        # XOR k times == XOR once if k is odd
                        tok = _word_tokens.get(w) if _word_tokens else _token64(w)
                        next_rz ^= tok

                    key = (h, rk, move_num, next_rz)
                    new_score = cached_board_score(board_to_tuple(b2), board_to_tuple(original_bonus))
                    prev = transpo.get(key)
                    if prev is not None and prev >= new_score:
                        continue  # dominated
                    transpo[key] = new_score
                    # size cap (simple trim)
                    if len(transpo) > transpo_cap:
                        # drop ~1% oldest-ish items (not true LRU, but cheap)
                        for _ in range(max(1, transpo_cap // 100)):
                            transpo.pop(next(iter(transpo)))

                # Compute features
                new_score = cached_board_score(board_to_tuple(b2), board_to_tuple(original_bonus))
                prem = premium_coverage_from_move(b, b2, original_bonus, w, r0, c0, d)
                mob = count_future_mobility(b2)

                # Diversity penalty: penalize repeated (row,col,dir) at this depth
                key_div = (r0, c0, d)
                penalty = gamma_diversity * repeat_counts[key_div]
                repeat_counts[key_div] += 1

                priority = new_score + alpha_premium * prem + beta_mobility * mob - penalty

                pl2 = {(r, c) for r in range(N) for c in range(N) if len(b2[r][c]) == 1}
                next_words = rem_words.copy()
                # remove all occurrences of w (mirrors earlier behaviour)
                try:
                    while True:
                        next_words.remove(w)
                except ValueError:
                    pass

                # Update remaining-words zobrist for child
                if use_transpo:
                    # we already computed odd/even above; recompute here to keep code clear
                    cnt = rem_words.count(w)
                    child_rz = rem_rz
                    if cnt % 2 == 1:
                        tok = _word_tokens.get(w)
                        if tok is None:
                            tok = _token64(w)
                            _word_tokens[w] = tok
                        child_rz ^= tok
                else:
                    child_rz = None

                next_state.append(
                    (
                        priority,
                        new_score,
                        b2,
                        rack_after,
                        pl2,
                        moves + [(sc, w, d, r0, c0)],
                        next_words,
                        child_rz,
                    )
                )

        vlog(f"full_beam_search move {move_num}: {len(state)} states expanded to {len(next_state)}", t0)
        # Sort by priority (desc), keep top beam_width states
        next_state.sort(key=lambda x: x[0], reverse=True)
        if beam_width is not None:
            next_state = next_state[:beam_width]
        state = next_state

        # Track best by *true* score (not the heuristic priority)
        if state and state[0][1] > best_score:
            best_score = state[0][1]
            best_board = state[0][2]
            best_moves = state[0][5]

        move_num += 1

    return best_score, best_board, best_moves


def beam_from_first(
    play,
    board,
    rack_count,
    words,
    wordset,
    original_bonus,
    beam_width,
    max_moves=20,
    prefixset=None,
    alpha_premium=ALPHA_PREMIUM_DEFAULT,
    beta_mobility=BETA_MOBILITY_DEFAULT,
    gamma_diversity=GAMMA_DIVERSITY_DEFAULT,
    use_transpo=False,
    transpo_cap=200000,
):
    """Run a beam search after making an initial play."""
    play_word = play[1]
    # Remove ALL occurrences of the played word from the simulation list
    words_for_sim = [w for w in words if w != play_word]

    # If using transpo, prebuild tokens and initial remaining-words zobrist for this branch
    word_tokens = {w: _token64(w) for w in set(words_for_sim)} if use_transpo else None
    rem_rz = None
    if use_transpo:
        rz = 0
        for w in words_for_sim:
            rz ^= word_tokens[w]
        rem_rz = rz

    board_copy = [row[:] for row in board]
    rack_count_copy = rack_count.copy()
    can_play, rack_after_first = can_play_word_on_board(
        play_word, play[3], play[4], play[2], board_copy, rack_count_copy, wordset, prefixset
    )
    if not can_play:
        return (float("-inf"), None, None)
    place_word(board_copy, play_word, play[3], play[4], play[2])
    if not validate_new_words(board_copy, wordset, play_word, play[3], play[4], play[2]):
        return (float("-inf"), None, None)
    placed_copy = {(r, c) for r in range(N) for c in range(N) if len(board_copy[r][c]) == 1}
    score, final_board, moves = full_beam_search(
        board_copy,
        rack_after_first,
        words_for_sim,
        wordset,
        prefixset,
        placed_copy,
        original_bonus,
        beam_width=beam_width,
        max_moves=max_moves,
        alpha_premium=alpha_premium,
        beta_mobility=beta_mobility,
        gamma_diversity=gamma_diversity,
        use_transpo=use_transpo,
        transpo_cap=transpo_cap,
        _word_tokens=word_tokens,
        _rem_zobrist=rem_rz,
    )
    return (score, final_board, [(play[0], play_word, play[2], play[3], play[4])] + (moves if moves else []))


def explore_alternatives(
    play,
    board,
    rack_count,
    pruned_words,
    wordset,
    original_bonus,
    beam_width,
    max_moves,
    prefixset=None,
    alpha_premium=ALPHA_PREMIUM_DEFAULT,
    beta_mobility=BETA_MOBILITY_DEFAULT,
    gamma_diversity=GAMMA_DIVERSITY_DEFAULT,
    use_transpo=False,
    transpo_cap=200000,
):
    """Helper function to explore alternative moves for a given start word."""
    score, board_result, moves = beam_from_first(
        play,
        board,
        rack_count,
        pruned_words,
        wordset,
        original_bonus,
        beam_width,
        max_moves,
        prefixset,
        alpha_premium,
        beta_mobility,
        gamma_diversity,
        use_transpo,
        transpo_cap,
    )
    if board_result and board_valid(board_result, wordset):
        return score, board_result, moves
    return None, None, None


def parallel_first_beam(
    board,
    rack,
    words,
    wordset,
    original_bonus,
    beam_width=5,
    num_games=100,
    first_moves=None,
    max_moves=20,
    prefixset=None,
    alpha_premium=ALPHA_PREMIUM_DEFAULT,
    beta_mobility=BETA_MOBILITY_DEFAULT,
    gamma_diversity=GAMMA_DIVERSITY_DEFAULT,
    use_transpo=False,
    transpo_cap=200000,
):
    """Search game states starting from multiple first moves in parallel."""
    rack_count = Counter(rack)
    if first_moves is None:
        first_moves = num_games
    t0 = time.time()
    pruned_words = prune_words(words, rack_count, board)
    log_with_time(f"Pruned word list: {len(pruned_words)} words", color=Fore.CYAN)
    vlog("Initial prune_words", t0)

    t1 = time.time()
    first_choices = find_best(
        board,
        rack_count,
        pruned_words,
        wordset,
        prefixset=prefixset,
        touch=None,
        original_bonus=original_bonus,
        top_k=first_moves,
    )
    vlog("find_best for first moves", t1)

    results = []
    best_total = float("-inf")
    best_results = []
    seen_best_boards = set()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_info = {
            executor.submit(
                explore_alternatives,
                play,
                board,
                rack_count,
                pruned_words,
                wordset,
                original_bonus,
                beam_width,
                max_moves,
                prefixset,
                alpha_premium,
                beta_mobility,
                gamma_diversity,
                use_transpo,
                transpo_cap,
            ): (time.time(), play, i)
            for i, play in enumerate(first_choices)
        }
        for future in concurrent.futures.as_completed(future_to_info):
            start, play, idx = future_to_info[future]
            score, board_result, moves = future.result()
            elapsed = time.time() - start
            _, word, direction, row, col = play
            status_msg = ""
            print_board_flag = False
            if moves is not None:
                results.append((score, board_result, moves))
                board_key = tuple(tuple(r) for r in board_result)
                if score > best_total:
                    best_total = score
                    best_results = [(score, board_result, moves)]
                    seen_best_boards = {board_key}
                    status_msg = " New High Score!"
                    print_board_flag = True
                elif score == best_total:
                    if board_key not in seen_best_boards:
                        seen_best_boards.add(board_key)
                        status_msg = " Equal high score."
                        print_board_flag = True
                        best_results.append((score, board_result, moves))
            msg_color = Fore.GREEN if status_msg else Fore.LIGHTBLUE_EX
            duration_msg = f" (duration: {elapsed:.3f}s)" if VERBOSE else ""
            log_with_time(
                f"Game {idx+1}/{len(first_choices)}: {word} at {row},{col},{direction.value} → final score: {score}{duration_msg}{status_msg}",
                color=msg_color,
            )
            if print_board_flag:
                with PRINT_LOCK:
                    if status_msg.strip() == "New High Score!":
                        print(f"\n{Fore.GREEN}New best score found: {score}{Style.RESET_ALL}", flush=True)
                    else:
                        print(f"\n{Fore.MAGENTA}Equal best score found: {score}{Style.RESET_ALL}", flush=True)
                print_board(board_result, original_bonus)
            vlog(f"beam_from_first {idx+1}", start)

    if not best_results:
        return 0, []

    return best_total, best_results