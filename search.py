from collections import Counter, defaultdict
import concurrent.futures
import time
import hashlib
from colorama import Fore, Style
from utils import log_with_time, vlog, N, PRINT_LOCK, VERBOSE, Direction
from board import board_valid, place_word, print_board
from score_cache import cached_board_score, board_to_tuple, board_hash

# ---- Default heuristic weights ----
ALPHA_PREMIUM_DEFAULT = 0.5
BETA_MOBILITY_DEFAULT = 0.2
GAMMA_DIVERSITY_DEFAULT = 0.01


def _token64(word: str) -> int:
    h = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little")


def _is_word(word, wordset=None, dawg=None):
    if dawg is not None:
        return dawg.is_word(word)
    if wordset is not None:
        return word in wordset
    return False


def _has_prefix(prefix, prefixset=None, wordset=None, dawg=None):
    if dawg is not None:
        return dawg.has_prefix(prefix)
    if prefixset is not None:
        return (prefix in prefixset) or (wordset is not None and prefix in wordset)
    return True


# ============== Helper utilities ==============
def _letters_from_rack(rack_count: Counter):
    for ch, n in rack_count.items():
        if n > 0 and len(ch) == 1 and "A" <= ch <= "Z":
            yield ch


def _board_is_empty(board):
    for r in range(N):
        for c in range(N):
            if len(board[r][c]) == 1:
                return False
    return True


def count_future_mobility(board):
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
    return tuple(sorted((ch, n) for ch, n in rack_counter.items() if n > 0))


# ============== Cross-checks (rack-aware) ==============
def compute_cross_checks_rack(board, rack_count, *, dawg=None, wordset=None):
    """Rack-limited cross-checks for perpendicular validity."""
    rack_letters = list(_letters_from_rack(rack_count))
    cross_across = [[None for _ in range(N)] for _ in range(N)]
    cross_down   = [[None for _ in range(N)] for _ in range(N)]

    # Vertical checks (used when placing across)
    for r in range(N):
        for c in range(N):
            cell = board[r][c]
            if len(cell) == 1:
                continue
            ru = r - 1
            while ru >= 0 and len(board[ru][c]) == 1:
                ru -= 1
            rd = r + 1
            while rd < N and len(board[rd][c]) == 1:
                rd += 1
            if rd - (ru + 1) <= 1:
                cross_across[r][c] = None
            else:
                above = [board[rr][c] for rr in range(ru + 1, r)]
                below = [board[rr][c] for rr in range(r + 1, rd)]
                allowed = set()
                for ch in rack_letters:
                    w = "".join(above) + ch + "".join(below)
                    if _is_word(w, wordset=wordset, dawg=dawg):
                        allowed.add(ch)
                cross_across[r][c] = allowed

    # Horizontal checks (used when placing down)
    for r in range(N):
        for c in range(N):
            cell = board[r][c]
            if len(cell) == 1:
                continue
            cl = c - 1
            while cl >= 0 and len(board[r][cl]) == 1:
                cl -= 1
            cr = c + 1
            while cr < N and len(board[r][cr]) == 1:
                cr += 1
            if cr - (cl + 1) <= 1:
                cross_down[r][c] = None
            else:
                left = [board[r][cc] for cc in range(cl + 1, c)]
                right = [board[r][cc] for cc in range(c + 1, cr)]
                allowed = set()
                for ch in rack_letters:
                    w = "".join(left) + ch + "".join(right)
                    if _is_word(w, wordset=wordset, dawg=dawg):
                        allowed.add(ch)
                cross_down[r][c] = allowed

    return cross_across, cross_down


# ============== Legacy scan path (kept) ==============
def prune_words(words, rack_count, board):
    t0 = time.time()
    from collections import Counter
    board_letters = Counter(cell for row in board for cell in row if len(cell) == 1)
    rack_plus_board = rack_count + board_letters
    pruned = []
    for w in words:
        wc = Counter(w)
        if all(rack_plus_board[ch] >= wc[ch] for ch in wc):
            pruned.append(w)
    vlog(f"prune_words: reduced from {len(words)} to {len(pruned)}", t0)
    return pruned


def can_play_word_on_board(word, r0, c0, d, board, rack, wordset=None, prefixset=None, *, dawg=None):
    """Lightweight verifier used by both paths."""
    rack = rack.copy()

    # compute fixed prefix leading into start for pruning
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
        if d == Direction.ACROSS:
            ru = r - 1
            while ru >= 0 and len(board[ru][c]) == 1:
                ru -= 1
            rd = r + 1
            while rd < N and len(board[rd][c]) == 1:
                rd += 1
            if rd - (ru + 1) > 1:
                parts = [board[rr][c] for rr in range(ru + 1, r)]
                parts.append(ch)
                parts += [board[rr][c] for rr in range(r + 1, rd)]
                w = "".join(parts)
                if len(w) > 1:
                    return _is_word(w, wordset=wordset, dawg=dawg)
        else:
            cl = c - 1
            while cl >= 0 and len(board[r][cl]) == 1:
                cl -= 1
            cr = c + 1
            while cr < N and len(board[r][cr]) == 1:
                cr += 1
            if cr - (cl + 1) > 1:
                parts = [board[r][cc] for cc in range(cl + 1, c)]
                parts.append(ch)
                parts += [board[r][cc] for cc in range(c + 1, cr)]
                w = "".join(parts)
                if len(w) > 1:
                    return _is_word(w, wordset=wordset, dawg=dawg)
        return True

    placed_any = False
    for i, ch in enumerate(word):
        r = r0 + (i if d == Direction.DOWN else 0)
        c = c0 + (i if d == Direction.ACROSS else 0)
        cell = board[r][c]

        if len(cell) == 1:
            if cell != ch:
                return False, None
        else:
            placed_any = True
            if rack.get(ch, 0) <= 0:
                return False, None
            if not check_cross(r, c, ch):
                return False, None
            rack[ch] -= 1

        prefix = existing_prefix + word[: i + 1]
        if len(prefix) >= 2 and not _has_prefix(prefix, prefixset=prefixset, wordset=wordset, dawg=dawg):
            return False, None

    if not placed_any:
        return False, None
    return True, rack


def is_valid_placement(w, board, rack_count, wordset, r0, c0, d, *, dawg=None):
    needed = {}
    placed_any = False
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
            placed_any = True
            cnt = needed.get(ch, 0) + 1
            if rack_count[ch] < cnt:
                return False
            needed[ch] = cnt
    return placed_any


def validate_new_words(board, wordset, w, r0, c0, d, *, dawg=None):
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
    if len(main_word) > 1 and not _is_word(main_word, wordset=wordset, dawg=dawg):
        return False

    for i, _ in enumerate(w):
        r = r0 + (i if d == Direction.DOWN else 0)
        c = c0 + (i if d == Direction.ACROSS else 0)
        if d == Direction.ACROSS:
            i0 = r
            while i0 > 0 and len(board[i0 - 1][c]) == 1:
                i0 -= 1
            i1 = r + 1
            while i1 < N and len(board[i1][c]) == 1:
                i1 += 1
            if i1 - i0 > 1:
                perp = "".join(board[ii][c] for ii in range(i0, i1))
                if not _is_word(perp, wordset=wordset, dawg=dawg):
                    return False
        else:
            j0 = c
            while j0 > 0 and len(board[r][j0 - 1]) == 1:
                j0 -= 1
            j1 = c + 1
            while j1 < N and len(board[r][j1]) == 1:
                j1 += 1
            if j1 - j0 > 1:
                perp = "".join(board[r][jj] for jj in range(j0, j1))
                if not _is_word(perp, wordset=wordset, dawg=dawg):
                    return False
    return True


def find_best(board, rack_count, words, wordset, prefixset=None, touch=None, original_bonus=None, top_k=10, *, dawg=None):
    """Legacy scan path kept for A/B."""
    t0 = time.time()
    checked = 0
    candidates = []
    for w in words:
        L = len(w)
        # across
        for r in range(N):
            for c in range(N - L + 1):
                if not is_valid_placement(w, board, rack_count, wordset, r, c, Direction.ACROSS, dawg=dawg):
                    continue
                if touch:
                    coords = [(r, c + i) for i in range(L)]
                    if not any(
                        (nr, nc) in touch
                        or any(abs(nr - tr) + abs(nc - tc) == 1 for (tr, tc) in touch)
                        for nr, nc in coords
                    ):
                        continue
                ok, _ = can_play_word_on_board(w, r, c, Direction.ACROSS, board, rack_count, wordset, prefixset, dawg=dawg)
                if not ok:
                    continue
                b2 = [row[:] for row in board]
                place_word(b2, w, r, c, Direction.ACROSS)
                if not validate_new_words(b2, wordset, w, r, c, Direction.ACROSS, dawg=dawg):
                    continue
                sc = cached_board_score(board_to_tuple(b2), board_to_tuple(original_bonus))
                candidates.append((sc, 0, L, w, Direction.ACROSS, r, c))
                checked += 1
        # down
        for r in range(N - L + 1):
            for c in range(N):
                if not is_valid_placement(w, board, rack_count, wordset, r, c, Direction.DOWN, dawg=dawg):
                    continue
                if touch:
                    coords = [(r + i, c) for i in range(L)]
                    if not any(
                        (nr, nc) in touch
                        or any(abs(nr - tr) + abs(nc - tc) == 1 for (tr, tc) in touch)
                        for nr, nc in coords
                    ):
                        continue
                ok, _ = can_play_word_on_board(w, r, c, Direction.DOWN, board, rack_count, wordset, prefixset, dawg=dawg)
                if not ok:
                    continue
                b2 = [row[:] for row in board]
                place_word(b2, w, r, c, Direction.DOWN)
                if not validate_new_words(b2, wordset, w, r, c, Direction.DOWN, dawg=dawg):
                    continue
                sc = cached_board_score(board_to_tuple(b2), board_to_tuple(original_bonus))
                candidates.append((sc, 0, L, w, Direction.DOWN, r, c))
                checked += 1

    candidates.sort(reverse=True)
    vlog(f"find_best checked {checked} placements for {len(words)} words, returning top {top_k if top_k is not None else 'all'}", t0)
    if not candidates:
        return []
    selected = candidates if top_k is None else candidates[:top_k]
    return [(sc, w, d, r, c) for sc, _, _, w, d, r, c in selected]


# ============== Anchor + DAWG generator (now with empty-board emit cap) ==============
def _collect_anchors(board):
    anchors_across = set()
    anchors_down = set()
    any_letters = any(len(board[r][c]) == 1 for r in range(N) for c in range(N))
    if not any_letters:
        for r in range(N):
            for c in range(N):
                anchors_across.add((r, c))
                anchors_down.add((r, c))
        return anchors_across, anchors_down

    for r in range(N):
        for c in range(N):
            if len(board[r][c]) == 1:
                continue
            touching = (
                (r > 0 and len(board[r - 1][c]) == 1) or
                (r + 1 < N and len(board[r + 1][c]) == 1) or
                (c > 0 and len(board[r][c - 1]) == 1) or
                (c + 1 < N and len(board[r][c + 1]) == 1)
            )
            if not touching:
                continue
            anchors_across.add((r, c))
            anchors_down.add((r, c))
    return anchors_across, anchors_down


def _segment_limits(board, r, c, d):
    return (0, N)


def _cell_letter(board, r, c):
    cell = board[r][c]
    return cell if len(cell) == 1 else None


def _dfs_grow_from_start(
    board, start_r, start_c, d, segment_len, dawg, rack_count, wordset, original_bonus,
    anchor_abs_index, cross_across, cross_down, results, top_k, emit_budget
):
    xcheck = cross_across if d == Direction.ACROSS else cross_down

    def coord_at(i):
        return (start_r, start_c + i) if d == Direction.ACROSS else (start_r + i, start_c)

    rack = rack_count.copy()
    letters = []

    def in_bounds(i):
        r, c = coord_at(i)
        return 0 <= r < N and 0 <= c < N

    def try_emit():
        if emit_budget is not None and emit_budget[0] <= 0:
            return
        if len(letters) >= 2:
            w = "".join(letters)
            placed_any = any(len(board[coord_at(i)[0]][coord_at(i)[1]]) != 1 for i in range(len(letters)))
            if not placed_any:
                return
            r0, c0 = coord_at(0)
            b2 = [row[:] for row in board]
            place_word(b2, w, r0, c0, d)
            if validate_new_words(b2, wordset, w, r0, c0, d, dawg=dawg):
                sc = cached_board_score(board_to_tuple(b2), board_to_tuple(original_bonus))
                results.append((sc, w, d, r0, c0))
                if emit_budget is not None:
                    emit_budget[0] -= 1

    def rec(i, passed_anchor):
        if emit_budget is not None and emit_budget[0] <= 0:
            return
        if not in_bounds(i):
            return
        r, c = coord_at(i)
        fixed = _cell_letter(board, r, c)
        must_cross_ok = xcheck[r][c]

        if fixed is not None:
            letters.append(fixed)
            prefix = "".join(letters)
            if dawg.has_prefix(prefix):
                if passed_anchor or (i == anchor_abs_index):
                    if dawg.is_word(prefix):
                        try_emit()
                rec(i + 1, passed_anchor or (i == anchor_abs_index))
            letters.pop()
            return

        if must_cross_ok is not None and not must_cross_ok:
            return
        for ch, cnt in list(rack.items()):
            if emit_budget is not None and emit_budget[0] <= 0:
                break
            if cnt <= 0:
                continue
            if must_cross_ok is not None and ch not in must_cross_ok:
                continue
            letters.append(ch)
            prefix = "".join(letters)
            if dawg.has_prefix(prefix):
                will_pass = passed_anchor or (i == anchor_abs_index)
                if will_pass and dawg.is_word(prefix):
                    try_emit()
                rack[ch] -= 1
                rec(i + 1, will_pass)
                rack[ch] += 1
            letters.pop()

    rec(0, False)


def find_best_anchor(
    board,
    rack_count,
    wordset,
    original_bonus,
    top_k,
    *,
    dawg,
    empty_emit_cap: int | None = None,   # NEW: cap number of emits on empty board
):
    t0 = time.time()
    anchors_across, anchors_down = _collect_anchors(board)
    cross_across, cross_down = compute_cross_checks_rack(board, rack_count, dawg=dawg, wordset=wordset)

    results = []
    empty = _board_is_empty(board)
    emit_budget = [empty_emit_cap] if empty and empty_emit_cap and empty_emit_cap > 0 else None

    # Across
    for (r, c_anchor) in anchors_across:
        if len(board[r][c_anchor]) == 1:
            continue
        s, e = _segment_limits(board, r, c_anchor, Direction.ACROSS)
        left_empty = 0
        cc = c_anchor - 1
        while cc >= s and len(board[r][cc]) != 1:
            left_empty += 1
            cc -= 1
        for shift in range(left_empty + 1):
            if emit_budget is not None and emit_budget[0] <= 0:
                break
            start_c = c_anchor - shift
            anchor_abs_index = c_anchor - start_c
            _dfs_grow_from_start(
                board, r, start_c, Direction.ACROSS, e - start_c, dawg, rack_count,
                wordset, original_bonus, anchor_abs_index, cross_across, cross_down, results, top_k, emit_budget
            )

    # Down
    for (r_anchor, c) in anchors_down:
        if emit_budget is not None and emit_budget[0] <= 0:
            break
        if len(board[r_anchor][c]) == 1:
            continue
        s, e = _segment_limits(board, r_anchor, c, Direction.DOWN)
        up_empty = 0
        rr = r_anchor - 1
        while rr >= s and len(board[rr][c]) != 1:
            up_empty += 1
            rr -= 1
        for shift in range(up_empty + 1):
            if emit_budget is not None and emit_budget[0] <= 0:
                break
            start_r = r_anchor - shift
            anchor_abs_index = r_anchor - start_r
            _dfs_grow_from_start(
                board, start_r, c, Direction.DOWN, e - start_r, dawg, rack_count,
                wordset, original_bonus, anchor_abs_index, cross_across, cross_down, results, top_k, emit_budget
            )

    # De-dup identical placements, keep best score
    best_by_key = {}
    for sc, w, d, r0, c0 in results:
        key = (w, d, r0, c0)
        if key not in best_by_key or sc > best_by_key[key][0]:
            best_by_key[key] = (sc, w, d, r0, c0)
    results = list(best_by_key.values())

    def tie_key(item):
        sc, w, d, r0, c0 = item
        h = hashlib.blake2b(f"{w}|{d.value}|{r0}|{c0}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "little")

    if not results:
        vlog("anchor-gen produced 0 candidates", t0)
        return []

    results.sort(key=lambda x: (-x[0], tie_key(x)))

    # Empty-board blending still keeps some long forms
    if empty and top_k is not None:
        LONG_MIN = 5
        reserve_long = max(3, top_k // 3)
        primary = results[: max(0, top_k - reserve_long)]
        long_pool = [r for r in results if len(r[1]) >= LONG_MIN]
        long_pool.sort(key=lambda x: (-len(x[1]), -x[0], tie_key(x)))
        seen = {(w, d, r0, c0) for _, w, d, r0, c0 in primary}
        picked_long = []
        for entry in long_pool:
            key = (entry[1], entry[2], entry[3], entry[4])
            if key in seen:
                continue
            picked_long.append(entry)
            seen.add(key)
            if len(picked_long) >= reserve_long:
                break
        blended = primary + picked_long
        msg = f"anchor-gen emitted {len(results)} (capped {empty_emit_cap}) → blended {len(blended)} (empty-board)"
        vlog(msg, t0)
        return blended[:top_k]
    else:
        out = results if top_k is None else results[:top_k]
        msg = f"anchor-gen emitted {len(results)} (cap {'n/a' if not empty_emit_cap else empty_emit_cap}) → returned {len(out)}"
        vlog(msg, t0)
        return out


# ============== Beam search (HYBRID optional pad; pad defaults OFF) ==============
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
    dawg=None,
    use_anchor_gen=False,
    legacy_pad_k: int = 0,
    legacy_pad_ratio: float = 0.0,
    empty_anchor_cap: int | None = None,   # NEW: pass through for recursive moves (no effect midgame)
):
    init_score = cached_board_score(board_to_tuple(board), board_to_tuple(original_bonus)) if words else 0
    state = [(init_score, init_score, board, rack_count, set(placed), [], words, _rem_zobrist)]
    best_score = 0
    best_board = None
    best_moves = None
    move_num = 1

    transpo = {} if use_transpo else None

    if not words:
        if board_valid(board, wordset):
            best_score = init_score
            return best_score, board, []

    while state and move_num <= max_moves:
        t0 = time.time()
        next_state = []
        repeat_counts = defaultdict(int)

        for _, cur_score, b, rc, pl, moves, rem_words, rem_rz in state:
            touch = None if not moves else pl

            if use_anchor_gen and dawg is not None:
                t_anchor = time.time()
                anchor_cands = find_best_anchor(
                    b, rc, wordset, original_bonus, top_k=beam_width, dawg=dawg,
                    empty_emit_cap=empty_anchor_cap if _board_is_empty(b) else None
                )
                t_anchor = time.time() - t_anchor

                # Optional tiny legacy pad (defaults OFF)
                pad_cap = 6
                bw = beam_width if beam_width is not None else 10
                pad_k = 0
                if legacy_pad_k > 0 or legacy_pad_ratio > 0.0:
                    pad_k = min(pad_cap, max(legacy_pad_k, int(bw * legacy_pad_ratio)))

                legacy_cands = []
                t_legacy = 0.0
                if pad_k > 0:
                    t_legacy = time.time()
                    pruned_words = prune_words(rem_words, rc, b)
                    legacy_cands = find_best(
                        b,
                        rc,
                        pruned_words,
                        wordset,
                        prefixset=prefixset,
                        touch=touch,
                        original_bonus=original_bonus,
                        top_k=pad_k,
                        dawg=dawg,
                    )
                    t_legacy = time.time() - t_legacy

                # Merge & trim
                best_by_key = {}
                for cand in anchor_cands + legacy_cands:
                    sc, w, d, r0, c0 = cand
                    key = (w, d, r0, c0)
                    if key not in best_by_key or sc > best_by_key[key][0]:
                        best_by_key[key] = (sc, w, d, r0, c0)
                candidates = list(best_by_key.values())
                candidates.sort(key=lambda x: x[0], reverse=True)
                if beam_width is not None:
                    candidates = candidates[:beam_width]

                vlog(
                    f"gen move#{move_num}: anchor {len(anchor_cands)} in {t_anchor*1000:.1f}ms"
                    + (f", legacy pad {len(legacy_cands)} in {t_legacy*1000:.1f}ms" if pad_k > 0 else "")
                    + f" -> using {len(candidates)}"
                )
            else:
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
                    dawg=dawg,
                )

            for sc, w, d, r0, c0 in candidates:
                ok, rack_after = can_play_word_on_board(w, r0, c0, d, b, rc, wordset, prefixset, dawg=dawg)
                if not ok:
                    continue

                b2 = [row[:] for row in b]
                place_word(b2, w, r0, c0, d)
                if not validate_new_words(b2, wordset, w, r0, c0, d, dawg=dawg):
                    continue

                if use_transpo:
                    h = board_hash(board_to_tuple(b2), board_to_tuple(original_bonus))
                    rk = _rack_key(rack_after)
                    count_removed = rem_words.count(w)
                    next_rz = rem_rz
                    if count_removed % 2 == 1:
                        tok = _word_tokens.get(w) if _word_tokens else _token64(w)
                        next_rz ^= tok
                    key = (h, rk, move_num, next_rz)
                    new_score = cached_board_score(board_to_tuple(b2), board_to_tuple(original_bonus))
                    prev = transpo.get(key)
                    if prev is not None and prev >= new_score:
                        continue
                    transpo[key] = new_score
                    if len(transpo) > transpo_cap:
                        for _ in range(max(1, transpo_cap // 100)):
                            transpo.pop(next(iter(transpo)))

                new_score = cached_board_score(board_to_tuple(b2), board_to_tuple(original_bonus))
                prem = premium_coverage_from_move(b, b2, original_bonus, w, r0, c0, d)
                mob = count_future_mobility(b2)

                key_div = (r0, c0, d)
                penalty = gamma_diversity * repeat_counts[key_div]
                repeat_counts[key_div] += 1

                priority = new_score + alpha_premium * prem + beta_mobility * mob - penalty

                pl2 = {(r, c) for r in range(N) for c in range(N) if len(b2[r][c]) == 1}
                next_words = rem_words.copy()
                try:
                    while True:
                        next_words.remove(w)
                except ValueError:
                    pass

                child_rz = None
                if use_transpo:
                    cnt = rem_words.count(w)
                    child_rz = rem_rz
                    if cnt % 2 == 1:
                        tok = _word_tokens.get(w)
                        if tok is None:
                            tok = _token64(w)
                            _word_tokens[w] = tok
                        child_rz ^= tok

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
        next_state.sort(key=lambda x: x[0], reverse=True)
        if beam_width is not None:
            next_state = next_state[:beam_width]
        state = next_state

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
    *,
    dawg=None,
    use_anchor_gen=False,
    legacy_pad_k: int = 0,
    legacy_pad_ratio: float = 0.0,
    empty_anchor_cap: int | None = None,
):
    play_word = play[1]
    words_for_sim = [w for w in words if w != play_word]

    word_tokens = {w: _token64(w) for w in set(words_for_sim)} if use_transpo else None
    rem_rz = None
    if use_transpo:
        rz = 0
        for w in words_for_sim:
            rz ^= word_tokens[w]
        rem_rz = rz

    board_copy = [row[:] for row in board]
    rack_count_copy = rack_count.copy()
    ok, rack_after_first = can_play_word_on_board(
        play_word, play[3], play[4], play[2], board_copy, rack_count_copy, wordset, prefixset, dawg=dawg
    )
    if not ok:
        return (float("-inf"), None, None)
    place_word(board_copy, play_word, play[3], play[4], play[2])
    if not validate_new_words(board_copy, wordset, play_word, play[3], play[4], play[2], dawg=dawg):
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
        dawg=dawg,
        use_anchor_gen=use_anchor_gen,
        legacy_pad_k=legacy_pad_k,
        legacy_pad_ratio=legacy_pad_ratio,
        empty_anchor_cap=empty_anchor_cap,
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
    *,
    dawg=None,
    use_anchor_gen=False,
    legacy_pad_k: int = 0,
    legacy_pad_ratio: float = 0.0,
    empty_anchor_cap: int | None = None,
):
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
        dawg=dawg,
        use_anchor_gen=use_anchor_gen,
        legacy_pad_k=legacy_pad_k,
        legacy_pad_ratio=legacy_pad_ratio,
        empty_anchor_cap=empty_anchor_cap,
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
    *,
    dawg=None,
    use_anchor_gen=False,
    legacy_pad_k: int = 0,
    legacy_pad_ratio: float = 0.0,
    empty_anchor_cap: int | None = None,   # NEW: CLI-plumbed
):
    rack_count = Counter(rack)
    if first_moves is None:
        first_moves = num_games
    t0 = time.time()

    pruned_words = prune_words(words, rack_count, board)
    log_with_time(f"Pruned word list: {len(pruned_words)} words", color=Fore.CYAN)
    vlog("Initial prune_words", t0)

    t1 = time.time()
    if use_anchor_gen and dawg is not None:
        first_choices = find_best_anchor(
            board, rack_count, wordset, original_bonus, top_k=first_moves, dawg=dawg,
            empty_emit_cap=empty_anchor_cap
        )
    else:
        first_choices = find_best(
            board,
            rack_count,
            pruned_words,
            wordset,
            prefixset=prefixset,
            touch=None,
            original_bonus=original_bonus,
            top_k=first_moves,
            dawg=dawg,
        )
    vlog("first move generation", t1)

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
                dawg=dawg,
                use_anchor_gen=use_anchor_gen,
                legacy_pad_k=legacy_pad_k,
                legacy_pad_ratio=legacy_pad_ratio,
                empty_anchor_cap=empty_anchor_cap,
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
                        print(f"\n{Fore.MAGENTA}Equal best score found: {score}{Style.RESET_ALL}")
                print_board(board_result, original_bonus)
            vlog(f"beam_from_first {idx+1}", start)

    if not best_results:
        return 0, []

    return best_total, best_results