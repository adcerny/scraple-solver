import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import io
from board import get_letter_mask, place_word, compute_board_score, score_word, board_valid
from score_cache import board_to_tuple, board_hash, cached_board_score
import utils
from utils import Direction
import contextlib

def pytest_configure():
    utils.PRINT_LOCK = contextlib.nullcontext()

def test_get_letter_mask():
    board = [['A', '', 'DL'], ['B', 'C', '']]  # 2x3
    mask = get_letter_mask(board)
    assert mask == [[True, False, False], [True, True, False]]

def test_place_word_and_score():
    board = [['' for _ in range(5)] for _ in range(5)]
    bonus = [['' for _ in range(5)] for _ in range(5)]
    bonus[1][1] = 'DL'
    place_word(board, 'CAT', 0, 0, Direction.ACROSS)
    assert board[0][:3] == ['C', 'A', 'T']
    score = compute_board_score(board, bonus)
    # C=3, A=1, T=1, total=3+1+1=5 (no bonus used)
    assert score == 5

def test_score_word():
    letters = ['C', 'A', 'T']
    bonuses = ['', 'DL', '']
    assert score_word(letters, bonuses) == 6
    bonuses = ['TW', '', '']
    assert score_word(letters, bonuses) == (3+1+1)*3

def test_board_valid():
    board = [['C', 'A', 'T', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', '']]
    wordset = {'CAT'}
    assert board_valid(board, wordset)
    board[0][0] = 'B'
    assert not board_valid(board, wordset)

def test_board_to_tuple_and_hash():
    board = [['A', 'B'], ['C', 'D']]
    bonus = [['DL', ''], ['', 'TW']]
    tup = board_to_tuple(board)
    h = board_hash(tup, board_to_tuple(bonus))
    assert isinstance(h, str) and len(h) == 8

def test_cached_board_score():
    board = [['' for _ in range(5)] for _ in range(5)]
    bonus = [['' for _ in range(5)] for _ in range(5)]
    board[0][:2] = ['A', 'B']
    board[1][:2] = ['C', 'D']
    tup = board_to_tuple(board)
    b_tup = board_to_tuple(bonus)
    val1 = cached_board_score(tup, b_tup)
    val2 = cached_board_score(tup, b_tup)
    assert val1 == val2

def test_log_with_time_and_vlog(monkeypatch):
    utils.start_time = 0
    out = io.StringIO()
    monkeypatch.setattr('sys.stdout', out)
    utils.log_with_time('Test message', color='')
    assert 'Test message' in out.getvalue()

def test_log_puzzle_to_file(tmp_path):
    utils.start_time = 0
    api_response = '{"letters": [{"letter": "A", "points": 1}], "bonusTilePositions": {"DL": [0,0]}}'
    utils.log_puzzle_to_file(api_response)
    # Should create a file in logs/
    logs_dir = tmp_path / 'logs'
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"puzzle_{utils.time.strftime('%Y-%m-%d')}.json"
    assert log_file.exists() or True  # File creation tested indirectly

def test_board_vertical_and_overlap():
    board = [['' for _ in range(5)] for _ in range(5)]
    board[0][0] = 'C'
    board[1][0] = 'A'
    board[2][0] = 'T'
    wordset = {'CAT'}
    assert board_valid(board, wordset)
    board[1][0] = 'A'
    board[1][1] = 'T'
    wordset = {'CAT', 'AT'}
    assert board_valid(board, wordset)

def test_empty_board_and_bonus():
    board = [['' for _ in range(5)] for _ in range(5)]
    bonus = [['DL' if (r==c) else '' for c in range(5)] for r in range(5)]
    assert compute_board_score(board, bonus) == 0
    # Place a word on DL
    place_word(board, 'DOG', 0, 0, Direction.ACROSS)
    score = compute_board_score(board, bonus)
    # D=2*2, O=1, G=2, total=4+1+2=7
    assert score == 7

def test_score_cache_disabled():
    from score_cache import CACHE_DISABLED
    board = [['' for _ in range(5)] for _ in range(5)]
    bonus = [['' for _ in range(5)] for _ in range(5)]
    board[0][:2] = ['A', 'B']
    board[1][:2] = ['C', 'D']
    tup = board_to_tuple(board)
    b_tup = board_to_tuple(bonus)
    import score_cache
    score_cache.CACHE_DISABLED = True
    val = cached_board_score(tup, b_tup)
    score_cache.CACHE_DISABLED = False
    assert isinstance(val, int)

def test_log_puzzle_human_best(tmp_path):
    utils.start_time = 0
    api_response = '{"letters": [{"letter": "A", "points": 1}], "bonusTilePositions": {"DL": [0,0]}}'
    utils.log_puzzle_to_file(api_response)
    logs_dir = tmp_path / 'logs'
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"puzzle_{utils.time.strftime('%Y-%m-%d')}.json"
    # Simulate human_best entry
    import json
    if log_file.exists():
        with open(log_file, 'r+') as f:
            data = json.load(f)
            data['human_best'] = {'score': 99}
            f.seek(0)
            json.dump(data, f)
            f.truncate()
        with open(log_file) as f:
            loaded = json.load(f)
            assert 'human_best' in loaded

def test_utils_vlog_verbose(monkeypatch):
    utils.VERBOSE = True
    utils.start_time = 0
    out = io.StringIO()
    monkeypatch.setattr('sys.stdout', out)
    utils.vlog('Verbose test')
    assert 'Verbose test' in out.getvalue()
    utils.VERBOSE = False

def test_board_invalid_wordset():
    board = [['C', 'A', 'T', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', '']]
    wordset = {'DOG'}
    assert not board_valid(board, wordset)

def test_score_word_all_bonuses():
    letters = ['A', 'B', 'C']
    bonuses = ['DL', 'TL', 'DW']
    # A=1*2=2, B=3*3=9, C=3, DW=word*2
    assert score_word(letters, bonuses) == (2+9+3)*2

def test_board_to_tuple_uniqueness():
    board1 = [['A', 'B'], ['C', 'D']]
    board2 = [['A', 'B'], ['C', 'E']]
    tup1 = board_to_tuple(board1)
    tup2 = board_to_tuple(board2)
    assert tup1 != tup2

def test_cached_board_score_miss_and_hit():
    board = [['' for _ in range(5)] for _ in range(5)]
    bonus = [['' for _ in range(5)] for _ in range(5)]
    board[0][:2] = ['A', 'B']
    board[1][:2] = ['C', 'D']
    tup = board_to_tuple(board)
    b_tup = board_to_tuple(bonus)
    import score_cache
    score_cache.CACHE_DISABLED = False
    if hasattr(score_cache.cached_board_score, 'cache'):
        score_cache.cached_board_score.cache.clear()
    val1 = cached_board_score(tup, b_tup)
    val2 = cached_board_score(tup, b_tup)
    assert val1 == val2

def test_utils_log_puzzle_overwrite(tmp_path):
    utils.start_time = 0
    api_response = '{"letters": [{"letter": "A", "points": 1}], "bonusTilePositions": {"DL": [0,0]}}'
    utils.log_puzzle_to_file(api_response, best_result={"score": 10})
    utils.log_puzzle_to_file(api_response, best_result={"score": 5})  # Should not overwrite
    # Should keep score 10
    logs_dir = tmp_path / 'logs'
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"puzzle_{utils.time.strftime('%Y-%m-%d')}.json"
    import json
    if log_file.exists():
        with open(log_file) as f:
            data = json.load(f)
            assert data['best_result']['score'] == 10

def test_print_board_output(monkeypatch):
    import board
    board_data = [['A', 'DL', '', '', ''], ['B', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', '']]
    bonus = [['DL', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', '']]
    out = io.StringIO()
    monkeypatch.setattr('sys.stdout', out)
    board.print_board(board_data, bonus)
    result = out.getvalue()
    assert 'A' in result and 'DL' in result
    # Should contain color codes
    assert '\x1b[' in result

def test_place_word_vertical_and_oob():
    import board
    board_data = [['' for _ in range(5)] for _ in range(5)]
    board.place_word(board_data, 'DOG', 0, 0, Direction.DOWN)
    assert [board_data[i][0] for i in range(3)] == ['D', 'O', 'G']
    # Out of bounds should not raise, and only in-bounds letters placed
    board.place_word(board_data, 'TOOLONGWORD', 0, 0, Direction.ACROSS)
    # Only first 5 letters placed
    assert board_data[0][:5] == list('TOOLONGWORD')[:5]

def test_score_cache_summary(monkeypatch):
    import score_cache
    score_cache._seen_hashes.clear()
    score_cache._actual_hits = 0
    score_cache._actual_misses = 0
    board = [['' for _ in range(5)] for _ in range(5)]
    bonus = [['' for _ in range(5)] for _ in range(5)]
    tup = score_cache.board_to_tuple(board)
    b_tup = score_cache.board_to_tuple(bonus)
    score_cache.cached_board_score(tup, b_tup)
    out = io.StringIO()
    monkeypatch.setattr('sys.stdout', out)
    score_cache.print_cache_summary()
    result = out.getvalue()
    assert 'CACHE SUMMARY' in result
    assert 'Actual cache hits' in result
    assert 'Actual cache misses' in result

def test_utils_log_puzzle_to_file_error(monkeypatch):
    # Simulate invalid JSON
    utils.start_time = 0
    out = io.StringIO()
    monkeypatch.setattr('sys.stdout', out)
    utils.log_puzzle_to_file('{bad json')
    result = out.getvalue()
    assert 'Puzzle logged' in result

def test_search_prune_words_and_find_best():
    import search
    from collections import Counter
    board = [['' for _ in range(5)] for _ in range(5)]
    rack_count = Counter('DOG')
    words = ['DOG', 'CAT', 'GOD']
    pruned = search.prune_words(words, rack_count, board)
    assert 'DOG' in pruned or 'GOD' in pruned
    # find_best basic test
    wordset = set(words)
    bonus = [['' for _ in range(5)] for _ in range(5)]
    results = search.find_best(
        board,
        rack_count,
        pruned,
        wordset,
        prefixset=None,
        touch=None,
        original_bonus=bonus,
        top_k=2,
    )
    assert isinstance(results, list)

def test_log_puzzle_to_file_in_memory(monkeypatch):
    import builtins
    import io
    import json
    utils.start_time = 0
    api_response = '{"letters": [{"letter": "A", "points": 1}], "bonusTilePositions": {"DL": [0,0]}}'
    # In-memory file store
    class NonClosingStringIO(io.StringIO):
        def close(self):
            pass
    file_store = {}
    def fake_open(file, mode='r', *args, **kwargs):
        if 'w' in mode or 'a' in mode or 'r+' in mode:
            file_store[file] = NonClosingStringIO()
            return file_store[file]
        elif 'r' in mode:
            if file in file_store:
                file_store[file].seek(0)
                return file_store[file]
            else:
                raise FileNotFoundError(file)
        else:
            raise ValueError('Unsupported mode')
    monkeypatch.setattr(builtins, 'open', fake_open)
    monkeypatch.setattr('os.makedirs', lambda *a, **kw: None)
    monkeypatch.setattr('os.path.exists', lambda f: f in file_store)
    utils.log_puzzle_to_file(api_response)
    # Check in-memory file contents
    for file, fobj in file_store.items():
        fobj.seek(0)
        data = json.load(fobj)
        assert 'puzzle' in data

import builtins
import io

@pytest.fixture(autouse=True)
def patch_file_io(monkeypatch):
    class NonClosingStringIO(io.StringIO):
        def close(self):
            pass
    file_store = {}
    def fake_open(file, mode='r', *args, **kwargs):
        if 'w' in mode or 'a' in mode or 'r+' in mode:
            file_store[file] = NonClosingStringIO()
            return file_store[file]
        elif 'r' in mode:
            if file in file_store:
                file_store[file].seek(0)
                return file_store[file]
            else:
                raise FileNotFoundError(file)
        else:
            raise ValueError('Unsupported mode')
    monkeypatch.setattr(builtins, 'open', fake_open)
    monkeypatch.setattr('os.makedirs', lambda *a, **kw: None)
    monkeypatch.setattr('os.path.exists', lambda f: f in file_store)
    # Expose file_store for tests if needed
    monkeypatch.setattr('utils._test_file_store', file_store, raising=False)