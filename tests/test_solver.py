import io
import time
import re
from contextlib import redirect_stdout
from collections import Counter
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

import search
import utils
from board import board_valid, compute_board_score

# Helper to count words on a board
def count_words(board):
    N = utils.N
    letter_mask = [[len(cell) == 1 for cell in row] for row in board]
    count = 0
    for r in range(N):
        c = 0
        while c < N:
            if letter_mask[r][c]:
                start = c
                while c < N and letter_mask[r][c]:
                    c += 1
                if c - start >= 2:
                    count += 1
            else:
                c += 1
    for c in range(N):
        r = 0
        while r < N:
            if letter_mask[r][c]:
                start = r
                while r < N and letter_mask[r][c]:
                    r += 1
                if r - start >= 2:
                    count += 1
            else:
                r += 1
    return count

# Fixture to run a deterministic game search
@pytest.fixture
def run_games(monkeypatch):
    class DummyFuture:
        def __init__(self, result):
            self._result = result
        def result(self):
            return self._result

    class DummyExecutor:
        def __init__(self):
            self.futures = []
        def submit(self, fn, *args, **kwargs):
            fut = DummyFuture(fn(*args, **kwargs))
            self.futures.append(fut)
            return fut
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass

    def dummy_as_completed(fs):
        for f in fs:
            yield f

    monkeypatch.setattr(search.concurrent.futures, 'ProcessPoolExecutor', lambda: DummyExecutor())
    monkeypatch.setattr(search.concurrent.futures, 'as_completed', dummy_as_completed)

    orig_find_best = search.find_best

    def stub_find_best(board, rack_count, words, wordset, touch=None, original_bonus=None, top_k=10):
        if not getattr(stub_find_best, 'called', False):
            stub_find_best.called = True
            return [
                (5, 'HI', 'V', 0, 0),
                (2, 'IT', 'H', 1, 0),
                (5, 'HI', 'H', 1, 0),
            ]
        return orig_find_best(board, rack_count, words, wordset, touch, original_bonus, top_k)

    monkeypatch.setattr(search, 'find_best', stub_find_best)

    def _run():
        board = [['' for _ in range(utils.N)] for _ in range(utils.N)]
        bonus = [row[:] for row in board]
        words = ['HI', 'IT']
        wordset = set(words)
        rack = ['H', 'I', 'I', 'T']
        utils.start_time = time.time()
        utils.VERBOSE = False
        buf = io.StringIO()
        with redirect_stdout(buf):
            best_score, best_results = search.parallel_first_beam(
                board,
                rack,
                words,
                wordset,
                bonus,
                beam_width=10,
                num_games=3,
                first_moves=3,
                max_moves=2,
            )
        output = buf.getvalue()
        return best_score, best_results, output, bonus, wordset

    return _run


def test_solutions_valid_and_scored(run_games):
    best_score, results, output, bonus, wordset = run_games()
    assert best_score == 14
    assert len(results) == 2
    for score, board, moves in results:
        assert board_valid(board, wordset)
        assert count_words(board) >= 2
        assert compute_board_score(board, bonus) == score


def test_output_and_dedup(run_games):
    best_score, results, output, bonus, wordset = run_games()
    ansi = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
    clean = ansi.sub('', output)
    assert 'New best score found: 14' in clean
    assert 'Equal best score found: 14' in clean
    # Board should be printed immediately after new high score message
    idx_new = clean.index('New best score found: 14')
    idx_game2 = clean.index('Game 2/3')
    assert idx_new < idx_game2
    assert clean.index(' H  I', idx_new) < idx_game2
    # Exactly two board outputs
    assert clean.count(' H  I') == 2
    # Results list should only keep unique boards
    assert len(results) == 2
