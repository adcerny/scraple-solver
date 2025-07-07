import os
import json
import pytest
import time
from search import parallel_first_beam
from utils import N, MAPPING, start_time
from board import print_board

def load_puzzle_log(log_path):
    with open(log_path, 'r') as f:
        return json.load(f)

def reconstruct_board_and_rack(puzzle):
    board = [['' for _ in range(N)] for _ in range(N)]
    for bonus, pos in puzzle['bonusTilePositions'].items():
        if isinstance(pos[0], int):
            r, c = pos
            board[r][c] = MAPPING[bonus]
    rack = [t['letter'].upper() for t in puzzle['letters']]
    return board, rack

def test_solver_matches_or_beats_human_best(
    beam_width=10,
    first_moves=10,
    max_moves=20,
    log_path=os.path.join('logs', 'puzzle_2025-07-07.json')
):
    import utils
    utils.start_time = time.time()
    data = load_puzzle_log(log_path)
    puzzle = data['puzzle']
    human_best = data['human_best']
    words = []
    wordset = set()
    try:
        from solver import load_dictionary
        words, wordset = load_dictionary()
    except Exception:
        pass
    board, rack = reconstruct_board_and_rack(puzzle)
    original_bonus = [row[:] for row in board]
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
    assert best_total >= human_best['score'], f"Solver did not match or beat human best: {best_total} < {human_best['score']}\nBoard:\n{best_results[0][1]}"
