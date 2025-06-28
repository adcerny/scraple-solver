# --- board.py ---

from colorama import Fore, Style
from utils import N, LETTER_SCORES, log_with_time, vlog

def is_letter(cell):
    return len(cell) == 1

def print_board(board):
    for row in board:
        line = []
        for cell in row:
            if cell == 'DL':
                line.append(Fore.BLUE + 'DL' + Style.RESET_ALL)
            elif cell == 'TL':
                line.append(Fore.CYAN + 'TL' + Style.RESET_ALL)
            elif cell == 'DW':
                line.append(Fore.MAGENTA + 'DW' + Style.RESET_ALL)
            elif cell == 'TW':
                line.append(Fore.RED + 'TW' + Style.RESET_ALL)
            elif is_letter(cell):
                line.append(Fore.GREEN + f' {cell}' + Style.RESET_ALL)
            else:
                line.append(Style.DIM + '··' + Style.RESET_ALL)
        print(' '.join(line))
    print()

def place_word(board, w, r0, c0, d):
    for i, ch in enumerate(w):
        r = r0 + (i if d == 'V' else 0)
        c = c0 + (i if d == 'H' else 0)
        board[r][c] = ch

def compute_board_score(board, original_bonus):
    total = 0
    # Horizontal words
    for r in range(N):
        c = 0
        while c < N:
            if is_letter(board[r][c]):
                start = c
                while c < N and is_letter(board[r][c]):
                    c += 1
                if c - start >= 2:
                    total += score_word(board[r][start:c], original_bonus[r][start:c])
            else:
                c += 1
    # Vertical words
    for c in range(N):
        r = 0
        while r < N:
            if is_letter(board[r][c]):
                start = r
                while r < N and is_letter(board[r][c]):
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
    # Horizontal
    for r in range(N):
        c = 0
        while c < N:
            if is_letter(board[r][c]):
                start = c
                while c < N and is_letter(board[r][c]):
                    c += 1
                if c - start >= 2:
                    word = ''.join(board[r][start:c])
                    if word not in wordset:
                        return False
            else:
                c += 1
    # Vertical
    for c in range(N):
        r = 0
        while r < N:
            if is_letter(board[r][c]):
                start = r
                while r < N and is_letter(board[r][c]):
                    r += 1
                if r - start >= 2:
                    word = ''.join(board[i][c] for i in range(start, r))
                    if word not in wordset:
                        return False
            else:
                r += 1
    return True