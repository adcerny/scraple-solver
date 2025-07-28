import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from board import board_valid
from utils import N

def make_board(words, positions, direction):
    board = [['' for _ in range(N)] for _ in range(N)]
    for word, (r, c) in zip(words, positions):
        for i, ch in enumerate(word):
            if direction == 'H':
                board[r][c + i] = ch
            else:
                board[r + i][c] = ch
    return board


def test_single_word_valid():
    board = make_board(['CAT'], [(0, 0)], 'H')
    wordset = {'CAT'}
    assert board_valid(board, wordset)


def test_two_connected_words_valid():
    board = make_board(['CAT', 'AT'], [(0, 0), (0, 1)], 'H')
    wordset = {'CAT', 'AT'}
    assert board_valid(board, wordset)


def test_two_disconnected_words_invalid():
    board = make_board(['CAT', 'DOG'], [(0, 0), (2, 0)], 'H')
    wordset = {'CAT', 'DOG'}
    assert not board_valid(board, wordset)


def test_word_not_in_wordset_invalid():
    board = make_board(['CAT'], [(0, 0)], 'H')
    wordset = {'DOG'}
    assert not board_valid(board, wordset)


def test_vertical_and_horizontal_connected():
    board = [['' for _ in range(N)] for _ in range(N)]
    # Place 'CAT' horizontally at (0,0)
    for i, ch in enumerate('CAT'):
        board[0][i] = ch
    # Place 'AT' vertically at (0,1)
    for i, ch in enumerate('AT'):
        board[i][1] = ch
    wordset = {'CAT', 'AT'}
    assert board_valid(board, wordset)


def test_vertical_and_horizontal_disconnected():
    board = [['' for _ in range(N)] for _ in range(N)]
    # Place 'CAT' horizontally at (0,0)
    for i, ch in enumerate('CAT'):
        board[0][i] = ch
    # Place 'DOG' vertically at (2,3)
    for i, ch in enumerate('DOG'):
        board[i+2][3] = ch
    wordset = {'CAT', 'DOG'}
    assert not board_valid(board, wordset)
