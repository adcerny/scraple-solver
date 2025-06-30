# --- exact_solver.py ---
from collections import Counter
from typing import List, Tuple, Optional
from pulp import LpProblem, LpVariable, LpBinary, LpMaximize, lpSum, PULP_CBC_CMD

from utils import N
from board import score_word, board_valid, compute_board_score


def solve_ilp(board: List[List[str]], rack: List[str], words: List[str], wordset: set, original_bonus: List[List[str]]) -> Tuple[Optional[List[List[str]]], int]:
    """Solve the puzzle exactly using integer programming.

    Returns a tuple of (best_board, score). If no solution exists, best_board is None.
    """
    rack_count = Counter(rack)
    pred_letters = {}
    for r in range(N):
        for c in range(N):
            cell = board[r][c]
            if len(cell) == 1:
                pred_letters[(r, c)] = cell

    pred_count = Counter(pred_letters.values())

    words_n = [w for w in words if len(w) == N]

    # Filter words that cannot be formed with the available letters at all.
    total_available = rack_count + pred_count
    words_n = [w for w in words_n
               if all(Counter(w)[ch] <= total_available[ch] for ch in Counter(w))]

    row_cands = []
    for r in range(N):
        pattern = [pred_letters.get((r, c)) for c in range(N)]
        cand = []
        for w in words_n:
            if all(p == ch or p is None for p, ch in zip(pattern, w)):
                cand.append(w)
        if not cand:
            return None, 0
        row_cands.append(cand)

    col_cands = []
    for c in range(N):
        pattern = [pred_letters.get((r, c)) for r in range(N)]
        cand = []
        for w in words_n:
            if all(p == ch or p is None for p, ch in zip(pattern, w)):
                cand.append(w)
        if not cand:
            return None, 0
        col_cands.append(cand)

    # Basic constraint propagation to prune inconsistent row/column words
    changed = True
    while changed:
        changed = False
        for r in range(N):
            for c in range(N):
                row_letters = {w[c] for w in row_cands[r]}
                col_letters = {w[r] for w in col_cands[c]}
                allowed = row_letters & col_letters
                if len(allowed) == 0:
                    return None, 0
                new_row = [w for w in row_cands[r] if w[c] in allowed]
                if len(new_row) != len(row_cands[r]):
                    row_cands[r] = new_row
                    changed = True
                new_col = [w for w in col_cands[c] if w[r] in allowed]
                if len(new_col) != len(col_cands[c]):
                    col_cands[c] = new_col
                    changed = True
                if not row_cands[r] or not col_cands[c]:
                    return None, 0

    row_scores = {}
    for r, cand in enumerate(row_cands):
        bonuses = [original_bonus[r][c] for c in range(N)]
        for w in cand:
            row_scores[(r, w)] = score_word(list(w), bonuses)

    col_scores = {}
    for c, cand in enumerate(col_cands):
        bonuses = [original_bonus[r][c] for r in range(N)]
        for w in cand:
            col_scores[(c, w)] = score_word(list(w), bonuses)

    prob = LpProblem("ScrapleExact", LpMaximize)

    row_vars = {(r, w): LpVariable(f"r_{r}_{w}", cat=LpBinary) for r, cand in enumerate(row_cands) for w in cand}
    col_vars = {(c, w): LpVariable(f"c_{c}_{w}", cat=LpBinary) for c, cand in enumerate(col_cands) for w in cand}

    for r, cand in enumerate(row_cands):
        prob += lpSum(row_vars[(r, w)] for w in cand) == 1
    for c, cand in enumerate(col_cands):
        prob += lpSum(col_vars[(c, w)] for w in cand) == 1

    letters = [chr(ord('A') + i) for i in range(26)]
    for r in range(N):
        for c in range(N):
            for L in letters:
                prob += lpSum(row_vars[(r, w)] for w in row_cands[r] if w[c] == L) == \
                        lpSum(col_vars[(c, w)] for w in col_cands[c] if w[r] == L)

    for L in letters:
        total = lpSum(row_vars[(r, w)] * w.count(L) for r, cand in enumerate(row_cands) for w in cand)
        prob += total <= pred_count[L] + rack_count[L]

    objective = lpSum(row_scores[(r, w)] * row_vars[(r, w)] for r, cand in enumerate(row_cands) for w in cand) + \
                lpSum(col_scores[(c, w)] * col_vars[(c, w)] for c, cand in enumerate(col_cands) for w in cand)
    prob += objective

    solver = PULP_CBC_CMD(msg=False)
    res = prob.solve(solver)

    if res != 1:
        return None, 0

    final_board = [['' for _ in range(N)] for _ in range(N)]
    for r, cand in enumerate(row_cands):
        for w in cand:
            if row_vars[(r, w)].value() == 1:
                for c, ch in enumerate(w):
                    final_board[r][c] = ch
                break

    score = compute_board_score([row[:] for row in final_board], original_bonus)
    if not board_valid(final_board, wordset):
        return None, 0
    return final_board, score
