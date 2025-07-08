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
    no_of_games=50,
    initial_positions=1,
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
    # Validate human board
    from board import board_valid, compute_board_score
    assert board_valid(human_best['final_board'], wordset), 'Human best board is not valid according to the dictionary.'
    actual_score = compute_board_score(human_best['final_board'], original_bonus)
    assert actual_score == human_best['score'], f'Human best board score mismatch: expected {human_best["score"]}, got {actual_score}.'
    # Try the top N initial positions for the first word, each with up to no_of_games first moves
    best_total = float('-inf')
    best_results = None
    for pos in range(1, initial_positions + 1):
        result_total, result_list = parallel_first_beam(
            board,
            rack,
            words,
            wordset,
            original_bonus,
            beam_width=beam_width,
            first_moves=no_of_games,
            max_moves=max_moves,
        )
        if result_total > best_total:
            best_total = result_total
            best_results = result_list
    assert best_total >= human_best['score'], f"Solver did not match or beat human best: {best_total} < {human_best['score']}\nBoard:\n{best_results[0][1]}"
