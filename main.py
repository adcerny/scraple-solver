from solver import run_solver
import multiprocessing
import sys

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_solver(sys.argv[1:])