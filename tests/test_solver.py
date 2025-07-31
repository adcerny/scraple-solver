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
from utils import Direction
from board import board_valid, compute_board_score

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
                (5, 'HI', Direction.DOWN, 0, 0),
                (2, 'IT', Direction.ACROSS, 1, 0),
                (5, 'HI', Direction.ACROSS, 1, 0),
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

def test_start_pos_happy(monkeypatch, capsys):
    """Test --start-word and --start-pos with a valid position."""
    import solver
    monkeypatch.setattr(sys, 'argv', [
        'solver.py',
        '--start-word', 'HI',
        '--start-pos', '0,0,D',
        '--depth', '2',
        '--beam-width', '2',
        '--num-games', '1',
        '--no-cache',
    ])
    monkeypatch.setattr(solver, 'fetch_board_and_rack', lambda: ([['' for _ in range(utils.N)] for _ in range(utils.N)], ['H', 'I', 'I', 'T'], None))
    monkeypatch.setattr(solver, 'load_dictionary', lambda: (['HI', 'IT'], set(['HI', 'IT']), ''))
    monkeypatch.setattr(solver, 'print_board', lambda board, bonus=None: None)
    solver.run_solver()
    out = capsys.readouterr().out
    assert "Best placement for 'HI':" in out
    assert "score" in out
    assert "position 0,0,D" in out


def test_start_pos_impossible(monkeypatch, capsys):
    """Test --start-word and --start-pos with an impossible position."""
    import solver
    monkeypatch.setattr(sys, 'argv', [
        'solver.py',
        '--start-word', 'HI',
        '--start-pos', '5,5,A',
        '--depth', '2',
        '--beam-width', '2',
        '--num-games', '1',
        '--no-cache',
    ])
    monkeypatch.setattr(solver, 'fetch_board_and_rack', lambda: ([['' for _ in range(utils.N)] for _ in range(utils.N)], ['H', 'I', 'I', 'T'], None))
    monkeypatch.setattr(solver, 'load_dictionary', lambda: (['HI', 'IT'], set(['HI', 'IT']), ''))
    monkeypatch.setattr(solver, 'print_board', lambda board, bonus=None: None)
    solver.run_solver()
    out = capsys.readouterr().out
    assert "Cannot place start word 'HI' at 5,5,A." in out


def test_start_pos_invalid_format(monkeypatch, capsys):
    """Test --start-pos with invalid format."""
    import solver
    monkeypatch.setattr(sys, 'argv', [
        'solver.py',
        '--start-word', 'HI',
        '--start-pos', 'badformat',
        '--depth', '2',
        '--beam-width', '2',
        '--num-games', '1',
        '--no-cache',
    ])
    monkeypatch.setattr(solver, 'fetch_board_and_rack', lambda: ([['' for _ in range(utils.N)] for _ in range(utils.N)], ['H', 'I', 'I', 'T'], None))
    monkeypatch.setattr(solver, 'load_dictionary', lambda: (['HI', 'IT'], set(['HI', 'IT']), ''))
    monkeypatch.setattr(solver, 'print_board', lambda board, bonus=None: None)
    solver.run_solver()
    out = capsys.readouterr().out
    assert "Invalid --start-word-pos format" in out


def test_improve_leaderboard(monkeypatch):
    import solver
    import board

    board_hs = [
        ['W', 'E', 'B'] + [''] * (utils.N - 3),
        ['I'] + [''] * (utils.N - 1),
        ['D'] + [''] * (utils.N - 1),
        ['E'] + [''] * (utils.N - 1),
    ] + [['' for _ in range(utils.N)] for _ in range(utils.N - 4)]
    bonus_hs = [['' for _ in range(utils.N)] for _ in range(utils.N)]
    leaderboard_data = {
        "scores": [
            {
                "score": 8,
                "gameState": {
                    "bonusTilePositions": {},
                    "placedTiles": {
                        "0-0": {"letter": "W"}, "0-1": {"letter": "E"}, "0-2": {"letter": "B"},
                        "1-0": {"letter": "I"}, "2-0": {"letter": "D"}, "3-0": {"letter": "E"}
                    },
                    "rack": ["X"]
                },
            }
        ]
    }

    class DummyFuture:
        def __init__(self, result):
            self._result = result
        def result(self):
            return self._result

    class DummyExecutor:
        def submit(self, fn, *args, **kwargs):
            return DummyFuture(fn(*args, **kwargs))
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(solver.concurrent.futures, 'ThreadPoolExecutor', lambda: DummyExecutor())

    def fake_requests_get(url, timeout=10):
        class Resp:
            def raise_for_status(self):
                pass
            def json(self):
                return leaderboard_data
        return Resp()

    monkeypatch.setattr(solver.requests, 'get', fake_requests_get)
    monkeypatch.setattr(solver, 'fetch_board_and_rack', lambda: ([['' for _ in range(utils.N)] for _ in range(utils.N)], [], None))
    monkeypatch.setattr(solver, 'load_dictionary', lambda: (['AB'], set(['AB']), ''))
    monkeypatch.setattr(solver, 'print_board', lambda *a, **k: None)
    monkeypatch.setattr(board, 'print_board', lambda *a, **k: None)
    monkeypatch.setattr(board, 'leaderboard_gamestate_to_board', lambda gs: (board_hs, bonus_hs))

    captured = []
    def fake_pfb(b, r, w, ws, ob, **kwargs):
        captured.append({'board': [row[:] for row in b], 'bonus': ob, 'rack': list(r)})
        return 0, []

    monkeypatch.setattr(solver, 'parallel_first_beam', fake_pfb)

    monkeypatch.setattr(sys, 'argv', ['solver.py', '--improve-leaderboard', '--num-games', '1', '--beam-width', '1', '--depth', '1', '--no-cache'])
    solver.run_solver()

    first_call = captured[0]
    assert first_call['bonus'] == bonus_hs
    assert first_call['board'][0][0] == 'W'
    assert first_call['board'][1][0] == 'I'
    assert first_call['board'][0][1] == ''
    assert first_call['board'][0][2] == ''
    assert Counter(first_call['rack']) == Counter(['E', 'B', 'X'])
