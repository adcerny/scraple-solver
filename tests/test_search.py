import pytest
from search import explore_alternatives
from board import board_valid
from utils import N, Direction
from collections import Counter
import random
import string

def test_explore_alternatives():
    # Mock data for the test
    board = [['' for _ in range(N)] for _ in range(N)]
    rack = ['A', 'B']  # Only enough tiles for 'AB'
    rack_count = Counter(rack)
    pruned_words = ['AB']
    wordset = set(pruned_words)
    original_bonus = [['' for _ in range(N)] for _ in range(N)]
    beam_width = 5
    max_moves = 10

    # Mock play (score, word, direction, row, col)
    play = (0, 'AB', Direction.ACROSS, 0, 0)

    # Call the function
    score, board_result, moves = explore_alternatives(
        play, board, rack_count, pruned_words, wordset, original_bonus, beam_width, max_moves
    )

    # Assertions
    assert score is not None, "Score should not be None"
    assert board_result is not None, "Board result should not be None"
    assert moves is not None, "Moves should not be None"
    assert board_valid(board_result, wordset), "Board result should be valid"
    assert len(moves) > 0, "Moves should not be empty"


def test_explore_alternatives_retries_invalid():
    # Setup a scenario where the first move leads to an invalid board, but a retry should succeed
    board = [['' for _ in range(N)] for _ in range(N)]
    rack = ['A', 'B']  # Only enough tiles for 'AB'
    rack_count = Counter(rack)
    pruned_words = ['AB']
    wordset = set(['AB'])  # Only 'AB' is valid
    original_bonus = [['' for _ in range(N)] for _ in range(N)]
    beam_width = 5
    max_moves = 10

    # First play is 'CD', which is not in wordset, so should be invalid
    play_invalid = (0, 'CD', Direction.ACROSS, 0, 0)
    score, board_result, moves = explore_alternatives(
        play_invalid, board, rack_count, pruned_words, wordset, original_bonus, beam_width, max_moves
    )
    # Should return None, None, None because 'CD' is not valid
    assert score is None
    assert board_result is None
    assert moves is None

    # Now try with a valid play
    play_valid = (0, 'AB', Direction.ACROSS, 0, 0)
    score2, board_result2, moves2 = explore_alternatives(
        play_valid, board, rack_count, pruned_words, wordset, original_bonus, beam_width, max_moves
    )
    assert score2 is not None
    assert board_result2 is not None
    assert moves2 is not None
    assert board_valid(board_result2, wordset)
    assert len(moves2) > 0


def test_board_valid_two_connected_words():
    from board import board_valid
    board = [['' for _ in range(N)] for _ in range(N)]
    # Place 'CAT' horizontally at (0,0)
    for i, ch in enumerate('CAT'):
        board[0][i] = ch
    # Place 'AT' horizontally at (0,1) (overlaps 'A' in 'CAT')
    for i, ch in enumerate('AT'):
        board[0][i+1] = ch
    wordset = {'CAT', 'AT'}
    assert board_valid(board, wordset)


def test_board_valid_two_disconnected_words():
    from board import board_valid
    board = [['' for _ in range(N)] for _ in range(N)]
    # Place 'CAT' horizontally at (0,0)
    for i, ch in enumerate('CAT'):
        board[0][i] = ch
    # Place 'DOG' horizontally at (2,0) (no overlap)
    for i, ch in enumerate('DOG'):
        board[2][i] = ch
    wordset = {'CAT', 'DOG'}
    assert not board_valid(board, wordset)


def test_board_valid_mixed_connected():
    from board import board_valid
    board = [['' for _ in range(N)] for _ in range(N)]
    # Place 'CAT' horizontally at (0,0)
    for i, ch in enumerate('CAT'):
        board[0][i] = ch
    # Place 'AT' vertically at (0,1) (overlaps 'A' in 'CAT')
    for i, ch in enumerate('AT'):
        board[i][1] = ch
    wordset = {'CAT', 'AT'}
    assert board_valid(board, wordset)


def test_board_valid_mixed_disconnected():
    from board import board_valid
    board = [['' for _ in range(N)] for _ in range(N)]
    # Place 'CAT' horizontally at (0,0)
    for i, ch in enumerate('CAT'):
        board[0][i] = ch
    # Place 'DOG' vertically at (2,3) (no overlap)
    for i, ch in enumerate('DOG'):
        board[i+2][3] = ch
    wordset = {'CAT', 'DOG'}
    assert not board_valid(board, wordset)


def test_solver_runs_with_random_board_and_dictionary():
    from search import parallel_first_beam
    from board import print_board
    # Generate a random dictionary of 100 words, length 2-5
    words = [''.join(random.choices(string.ascii_uppercase, k=random.randint(2, 5))) for _ in range(100)]
    wordset = set(words)
    # Generate a random rack of 5 letters
    rack = random.choices(string.ascii_uppercase, k=5)
    # Generate a random board with random bonus squares
    board = [['' for _ in range(N)] for _ in range(N)]
    bonus_types = ['DL', 'TL', 'DW', 'TW', '']
    for _ in range(random.randint(5, 10)):
        r = random.randint(0, N-1)
        c = random.randint(0, N-1)
        board[r][c] = random.choice(bonus_types[:-1])  # Don't fill with ''
    original_bonus = [row[:] for row in board]
    # Run the solver with high beam width and games
    try:
        best_total, best_results = parallel_first_beam(
            board,
            rack,
            words,
            wordset,
            original_bonus,
            beam_width=100,
            num_games=50,
            first_moves=50,
            max_moves=10,
        )
        # Just check that it runs and returns results
        assert isinstance(best_total, (int, float))
        assert isinstance(best_results, list)
    except Exception as e:
        pytest.fail(f"Solver raised an exception: {e}")


def test_solver_runs_with_random_start_word():
    from search import find_best, beam_from_first, prune_words
    words = [''.join(random.choices(string.ascii_uppercase, k=random.randint(2, 5))) for _ in range(100)]
    wordset = set(words)
    attempts = 0
    while attempts < 100:
        start_word = random.choice(words)
        rack = list(start_word)
        while len(rack) < 5:
            rack.append(random.choice(string.ascii_uppercase))
        random.shuffle(rack)
        rack_counter = Counter(rack)
        board = [['' for _ in range(N)] for _ in range(N)]
        bonus_types = ['DL', 'TL', 'DW', 'TW', '']
        for _ in range(random.randint(5, 10)):
            r = random.randint(0, N-1)
            c = random.randint(0, N-1)
            board[r][c] = random.choice(bonus_types[:-1])
        original_bonus = [row[:] for row in board]
        pruned_words = prune_words(words, rack_counter, board)
        valid_placements = find_best(
            board,
            rack_counter,
            [start_word],
            wordset,
            None,
            original_bonus,
            top_k=None
        )
        if valid_placements:
            best_placement = max(valid_placements, key=lambda x: x[0])
            score, board_after, moves = beam_from_first(
                best_placement,
                board,
                rack_counter,
                pruned_words,
                wordset,
                original_bonus,
                beam_width=200,
                max_moves=10
            )
            if board_after is not None:
                assert isinstance(score, (int, float))
                assert board_after is not None
                assert isinstance(moves, list)
                return
        attempts += 1
    pytest.fail("Could not find a start word and rack with a valid board after 100 attempts")
