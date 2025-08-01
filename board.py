def leaderboard_gamestate_to_board(game_state):
    """
    Convert a leaderboard API gameState dict to a 2D board (list of lists of str) for print_board.
    """
    N_local = N  # Use imported N
    board = [['' for _ in range(N_local)] for _ in range(N_local)]
    bonus_board = [['' for _ in range(N_local)] for _ in range(N_local)]
    # Place bonuses
    bonus_map = {
        'DOUBLE_LETTER': 'DL',
        'TRIPLE_LETTER': 'TL',
        'DOUBLE_WORD': 'DW',
        'TRIPLE_WORD': 'TW',
    }
    for bonus, pos in game_state.get('bonusTilePositions', {}).items():
        if isinstance(pos[0], int):
            r, c = pos
            bonus_code = bonus_map.get(bonus, '')
            bonus_board[r][c] = bonus_code
    # Place tiles
    for key, tile in game_state.get('placedTiles', {}).items():
        r, c = map(int, key.split('-'))
        letter = tile.get('letter')
        if letter:
            board[r][c] = letter.upper()
    # Fill empty cells with bonus codes for print_board coloring
    for r in range(N_local):
        for c in range(N_local):
            if not board[r][c]:
                board[r][c] = bonus_board[r][c] if bonus_board[r][c] else ''
    return board, bonus_board

from colorama import Fore, Style
from utils import N, LETTER_SCORES, log_with_time, vlog, PRINT_LOCK, Direction

# Helper: precompute a mask of letter positions for a board
# Returns a 2D list of bools: True if cell is a letter, else False
def get_letter_mask(board):
    return [[len(cell) == 1 for cell in row] for row in board]

def print_board(board, bonus=None):
    """Thread-safe printing of a board. If ``bonus`` is provided, it should
    represent the original bonus layout so that tiles can be colored based on
    the square they occupy."""
    with PRINT_LOCK:
        lines = []
        for r, row in enumerate(board):
            line = []
            for c, cell in enumerate(row):
                bonus_code = bonus[r][c] if bonus else ''
                if cell == 'DL':
                    line.append(Fore.CYAN + 'DL' + Style.RESET_ALL)
                elif cell == 'TL':
                    line.append(Fore.BLUE + 'TL' + Style.RESET_ALL)
                elif cell == 'DW':
                    line.append(Fore.MAGENTA + 'DW' + Style.RESET_ALL)
                elif cell == 'TW':
                    line.append(Fore.RED + 'TW' + Style.RESET_ALL)
                elif cell and len(cell) == 1:
                    color = {
                        'DL': Fore.CYAN,
                        'TL': Fore.BLUE,
                        'DW': Fore.MAGENTA,
                        'TW': Fore.RED
                    }.get(bonus_code, Fore.GREEN)
                    line.append(color + f' {cell}' + Style.RESET_ALL)
                else:
                    line.append(Style.DIM + '··' + Style.RESET_ALL)
            lines.append(' '.join(line))
        print('\n'.join(lines), flush=True)
        print(flush=True)

def place_word(board, w, r0, c0, d):
    """Place ``w`` on ``board`` starting at ``r0,c0`` in direction ``d``."""
    N = len(board)
    for i, ch in enumerate(w):
        r = r0 + (i if d == Direction.DOWN else 0)
        c = c0 + (i if d == Direction.ACROSS else 0)
        if 0 <= r < N and 0 <= c < N:
            board[r][c] = ch

def compute_board_score(board, original_bonus):
    total = 0
    letter_mask = get_letter_mask(board)
    # Horizontal words
    for r in range(N):
        c = 0
        while c < N:
            if letter_mask[r][c]:
                start = c
                while c < N and letter_mask[r][c]:
                    c += 1
                if c - start >= 2:
                    total += score_word(board[r][start:c], original_bonus[r][start:c])
            else:
                c += 1
    # Vertical words
    for c in range(N):
        r = 0
        while r < N:
            if letter_mask[r][c]:
                start = r
                while r < N and letter_mask[r][c]:
                    r += 1
                if r - start >= 2:
                    word = [board[i][c] for i in range(start, r)]
                    bonus = [original_bonus[i][c] for i in range(start, r)]
                    total += score_word(word, bonus)
            else:
                r += 1
    return total

def score_word(letters, bonuses):
    score, word_multiplier = 0, 1
    for ch, bonus in zip(letters, bonuses):
        val = LETTER_SCORES[ch]
        if bonus == 'DL': val *= 2
        elif bonus == 'TL': val *= 3
        elif bonus == 'DW': word_multiplier *= 2
        elif bonus == 'TW': word_multiplier *= 3
        score += val
    return score * word_multiplier

def board_valid(board, wordset):
    letter_mask = get_letter_mask(board)
    words = []
    word_positions = []
    # Horizontal
    for r in range(N):
        c = 0
        while c < N:
            if letter_mask[r][c]:
                start = c
                while c < N and letter_mask[r][c]:
                    c += 1
                if c - start >= 2:
                    word = ''.join(board[r][start:c])
                    if word not in wordset:
                        return False
                    words.append(word)
                    word_positions.append([(r, cc) for cc in range(start, c)])
            else:
                c += 1
    # Vertical
    for c in range(N):
        r = 0
        while r < N:
            if letter_mask[r][c]:
                start = r
                while r < N and letter_mask[r][c]:
                    r += 1
                if r - start >= 2:
                    word = ''.join(board[i][c] for i in range(start, r))
                    if word not in wordset:
                        return False
                    words.append(word)
                    word_positions.append([(rr, c) for rr in range(start, r)])
            else:
                r += 1
    if not words:
        return False
    if len(words) == 1:
        return True
    # Check connectivity: all word positions must be part of a single connected component
    from collections import deque
    visited = set()
    queue = deque(word_positions[0])
    while queue:
        pos = queue.popleft()
        if pos in visited:
            continue
        visited.add(pos)
        r, c = pos
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < N and 0 <= nc < N and letter_mask[nr][nc] and (nr, nc) not in visited:
                queue.append((nr, nc))
    # All positions in all words must be visited
    all_positions = set(pos for positions in word_positions for pos in positions)
    return all_positions.issubset(visited)
