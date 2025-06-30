import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from collections import Counter

from board import place_word, compute_board_score, board_valid
from utils import N

def empty_board():
    return [["" for _ in range(N)] for _ in range(N)]


def empty_bonus():
    return [["" for _ in range(N)] for _ in range(N)]


def test_compute_board_score_basic():
    board = empty_board()
    bonus = empty_bonus()
    place_word(board, "HELLO", 0, 0, 'H')
    bonus[0][1] = 'DL'
    bonus[0][4] = 'DW'
    assert compute_board_score(board, bonus) == 18


def test_board_valid_true():
    board = empty_board()
    place_word(board, "HELLO", 0, 0, 'H')
    assert board_valid(board, {"HELLO"})


def test_board_valid_false():
    board = empty_board()
    place_word(board, "HELXO", 0, 0, 'H')
    assert not board_valid(board, {"HELLO"})

