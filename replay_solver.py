import os
import json
from glob import glob
from utils import log_with_time
from colorama import Fore

def find_latest_log(logs_dir='logs'):
    files = glob(os.path.join(logs_dir, 'puzzle_*.json'))
    if not files:
        raise FileNotFoundError('No puzzle log files found.')
    return max(files, key=os.path.getmtime)

def load_log(log_path=None):
    if log_path is None:
        log_path = find_latest_log()
    with open(log_path, 'r') as f:
        return json.load(f), log_path

def print_board(board):
    for row in board:
        print(' '.join(cell if cell else '..' for cell in row))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Replay and compare solver and human best results.')
    parser.add_argument('--log', type=str, default=None, help='Path to puzzle log JSON (default: latest in logs/)')
    args = parser.parse_args()

    data, log_path = load_log(args.log)
    print(f"Loaded log: {log_path}")

    best = data.get('best_result', {})
    human = data.get('human_best', {})

    print(f"\nSolver best score: {best.get('score', 'N/A')}")
    if 'final_board' in best:
        print_board(best['final_board'])
    else:
        print('(No solver board found)')

    print(f"\nHuman best score: {human.get('score', 'N/A')}")
    if 'final_board' in human:
        print_board(human['final_board'])
    else:
        print('(No human board found)')

    if best.get('score', 0) < human.get('score', 0):
        print(f"\n{Fore.YELLOW}Human best is higher by {human['score'] - best['score']} points!{Fore.RESET}")
    elif best.get('score', 0) > human.get('score', 0):
        print(f"\n{Fore.GREEN}Solver best is higher by {best['score'] - human['score']} points!{Fore.RESET}")
    else:
        print(f"\n{Fore.CYAN}Scores are equal!{Fore.RESET}")
