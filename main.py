# --- main.py ---

from solver import run_solver
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_solver()
