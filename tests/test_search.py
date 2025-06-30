import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from collections import Counter

from search import can_play_word_on_board, is_valid_placement
from utils import N

def empty_board():
    return [["" for _ in range(N)] for _ in range(N)]


def test_can_play_word_success():
    board = empty_board()
    board[0][0] = 'H'
    can_play, remaining = can_play_word_on_board("HI", 0, 0, 'H', board, Counter({'I':1}))
    assert can_play
    assert remaining['I'] == 0


def test_can_play_word_fail():
    board = empty_board()
    can_play, _ = can_play_word_on_board("HELLO", 0, 0, 'H', board, Counter('HI'))
    assert not can_play


def test_is_valid_placement_cross_invalid():
    board = empty_board()
    board[0][1] = 'X'
    rack = Counter('HI')
    assert not is_valid_placement('HI', board, rack, {'HI'}, 0, 0, 'V')


def test_is_valid_placement_cross_valid():
    board = empty_board()
    board[0][1] = 'I'
    rack = Counter('HI')
    assert is_valid_placement('HI', board, rack, {'HI'}, 0, 0, 'V')


