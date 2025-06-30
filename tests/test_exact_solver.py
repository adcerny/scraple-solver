import pytest
from exact_solver import solve_ilp
from utils import N
from board import board_valid, compute_board_score


def test_word_square_solution():
    words = ['MAKES', 'ABOVE', 'KOREA', 'EVENT', 'SEATS']
    board = [['' for _ in range(N)] for _ in range(N)]
    bonuses = [['' for _ in range(N)] for _ in range(N)]
    rack = list(''.join(words))
    wordset = set(words)
    final_board, score = solve_ilp(board, rack, words, wordset, bonuses)
    assert final_board is not None
    assert board_valid(final_board, wordset)
    assert score == compute_board_score(final_board, bonuses)
    assert [''.join(row) for row in final_board] == words

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import pytest
    sys.exit(pytest.main([__file__]))
