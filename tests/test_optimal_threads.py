import subprocess
import time
import sys
import os

# Path to main entry point (relative to this script's directory)
MAIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'main.py')

best = min(results, key=lambda x: x[1])

# Use a fixed beam width for consistency
BEAM_WIDTH = 10
START_THREADS = 6 # starting thread count
MAX_THREADS = 50  # maximum thread count to test
REPEATS = 1  # Number of runs per thread count for averaging
GAMES_PER_THREAD = 10  # Set how many games per thread to run (can adjust for heavier/lighter workloads)

results = []

for threads in range(START_THREADS, MAX_THREADS + 1):
    num_games = threads * GAMES_PER_THREAD
    print(f"\n=== Testing with threads = {threads}, num_games = {num_games} ===")
    times = []
    for _ in range(REPEATS):
        cmd = [
            sys.executable, MAIN,
            '--beam-width', str(BEAM_WIDTH),
            '--num-threads', str(threads),
            '--num-games', str(num_games),
            '--no-cache',  # Disable cache for fair timing
        ]
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        print(f"\nRunning: {' '.join(str(x) for x in cmd)}")
        start = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        elapsed = time.time() - start
        print(f"Return code: {proc.returncode}")
        print(f"STDOUT:\n{proc.stdout}")
        print(f"STDERR:\n{proc.stderr}")
        times.append(elapsed)
        print(f"Threads: {threads}, Games: {num_games}, Time: {elapsed:.2f}s")
    avg_time = sum(times) / len(times)
    results.append((threads, num_games, avg_time))

best = min(results, key=lambda x: x[2])
print("\n=== THREAD PERFORMANCE REPORT ===")
print(f"Optimal thread count: {best[0]} (Games: {best[1]}, Avg Time: {best[2]:.2f}s)")
print("All timing results:")
for t, ng, tm in results:
    print(f"Threads: {t}, Games: {ng}, Avg Time: {tm:.2f}s")
