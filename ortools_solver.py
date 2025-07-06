from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpSolverSolutionCallback
from collections import Counter
from board import score_word
import utils
from collections import Counter


def run_ortools_solver(board, rack, words, wordset, original_bonus):
    """
    OR-Tools CP-SAT solver for full-game Scraple optimisation on an N×N board.
    Returns: best_total (int), best_results (list of (score, board, moves_list)).
    Each move: (placement_score, word, direction, row, col).
    """
    N = len(board)
    rack_count = Counter(rack)

    # Pre-filter words that can be formed from rack tiles
    valid_words = []
    for w in words:
        L = len(w)
        if 2 <= L <= N and all(Counter(w)[ch] <= rack_count[ch] for ch in set(w)):
            valid_words.append(w)

    # Generate all possible placements
    placements = []
    for w in valid_words:
        L = len(w)
        # Horizontal
        for r in range(N):
            for c in range(N - L + 1):
                # Board starts empty (no pre-placed letters), so any placement fits
                coords = [(r, c + i) for i in range(L)]
                placements.append({'word': w, 'dir': 'H', 'r': r, 'c': c, 'coords': coords})
        # Vertical
        for r in range(N - L + 1):
            for c in range(N):
                coords = [(r + i, c) for i in range(L)]
                placements.append({'word': w, 'dir': 'V', 'r': r, 'c': c, 'coords': coords})

    # Compute static score for each placement
    for pl in placements:
        w = pl['word']
        if pl['dir'] == 'H':
            bonus_list = [original_bonus[pl['r']][pl['c'] + i] for i in range(len(w))]
        else:
            bonus_list = [original_bonus[pl['r'] + i][pl['c']] for i in range(len(w))]
        pl['score'] = score_word(list(w), bonus_list)

    # Build CP-SAT model
    model = cp_model.CpModel()
    place_vars = [model.NewBoolVar(f"place_{i}_{pl['word']}_{pl['r']}_{pl['c']}_{pl['dir']}")
                  for i, pl in enumerate(placements)]

    all_letters = list(rack_count.keys())
    cell_letter = {}
    for i in range(N):
        for j in range(N):
            for l in all_letters:
                cell_letter[(i, j, l)] = model.NewBoolVar(f"cell_{i}_{j}_{l}")
            # At most one letter per cell
            model.Add(sum(cell_letter[(i, j, l)] for l in all_letters) <= 1)

    # Link placement -> cell_letter
    for idx, pl in enumerate(placements):
        pv = place_vars[idx]
        for k, (i, j) in enumerate(pl['coords']):
            l = pl['word'][k]
            model.Add(pv <= cell_letter[(i, j, l)])

    # Link cell_letter -> placement (no stray letters)
    for i in range(N):
        for j in range(N):
            for l in all_letters:
                cov = []
                for idx, pl in enumerate(placements):
                    for k, (ii, jj) in enumerate(pl['coords']):
                        if (ii, jj) == (i, j) and pl['word'][k] == l:
                            cov.append(place_vars[idx])
                # If letter l appears in cell (i,j), at least one placement must cover it
                if cov:
                    model.Add(sum(cov) >= cell_letter[(i, j, l)])
                else:
                    model.Add(cell_letter[(i, j, l)] == 0)

    # Prevent two placements in same direction from overlapping
    for i in range(N):
        for j in range(N):
            # horizontal overlap
            h_cov = [place_vars[idx] for idx, pl in enumerate(placements)
                     if pl['dir'] == 'H' and (i, j) in pl['coords']]
            if h_cov:
                model.Add(sum(h_cov) <= 1)
            # vertical overlap
            v_cov = [place_vars[idx] for idx, pl in enumerate(placements)
                     if pl['dir'] == 'V' and (i, j) in pl['coords']]
            if v_cov:
                model.Add(sum(v_cov) <= 1)

    # Letter supply constraints
    for l, cnt in rack_count.items():
        model.Add(sum(cell_letter[(i, j, l)] for i in range(N) for j in range(N)) <= cnt)

    # Objective: maximize sum of placement scores
    obj_terms = []
    for idx, pl in enumerate(placements):
        obj_terms.append(pl['score'] * place_vars[idx])
    model.Maximize(sum(obj_terms))

    # Solve for optimal score
    solver_max = cp_model.CpSolver()
    solver_max.parameters.max_time_in_seconds = 60
    solver_max.parameters.num_search_workers = 8
    status = solver_max.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return 0, []
    best_total = int(solver_max.ObjectiveValue())

    # Constrain model to optimal score for enumeration
    model.Add(sum(obj_terms) == best_total)

    # Enumerate all equally-high solutions
    class AllSolutions(CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.results = []

        def OnSolutionCallback(self):
            # Gather chosen placements
            chosen = [idx for idx, pv in enumerate(place_vars) if self.Value(pv)]
            # Reconstruct board and move list
            board_filled = [row.copy() for row in board]
            moves = []
            for idx in chosen:
                pl = placements[idx]
                moves.append((pl['score'], pl['word'], pl['dir'], pl['r'], pl['c']))
                for k, (i, j) in enumerate(pl['coords']):
                    board_filled[i][j] = pl['word'][k]
            self.results.append((best_total, board_filled, moves))

    callback = AllSolutions()
    solver_enum = cp_model.CpSolver()
    solver_enum.parameters.max_time_in_seconds = 60
    solver_enum.parameters.num_search_workers = 8
    solver_enum.SearchForAllSolutions(model, callback)

    return best_total, callback.results