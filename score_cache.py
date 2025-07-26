from board import compute_board_score
import hashlib
import utils


_seen_hashes = {}
_actual_hits = 0
_actual_misses = 0
CACHE_DISABLED = False

def board_to_tuple(board):
    return tuple(tuple(row) for row in board)

def board_hash(board_tuple, bonus_tuple):
    s = str(board_tuple) + str(bonus_tuple)
    return hashlib.md5(s.encode()).hexdigest()[:8]

def cached_board_score(board_tuple, bonus_tuple):
    global _actual_hits, _actual_misses
    if CACHE_DISABLED:
        # Always recompute, do not use or update cache or hit/miss counts
        return compute_board_score(board_tuple, bonus_tuple)
    if utils.VERBOSE:
        h = board_hash(board_tuple, bonus_tuple)
        count = _seen_hashes.get(h, 0)
        _seen_hashes[h] = count + 1
    if not hasattr(cached_board_score, 'cache'):
        cached_board_score.cache = {}
    cache = cached_board_score.cache
    key = (board_tuple, bonus_tuple)
    if key in cache:
        _actual_hits += 1
        return cache[key]
    else:
        _actual_misses += 1
        val = compute_board_score(board_tuple, bonus_tuple)
        cache[key] = val
        return val

def print_cache_summary():
    print(f"[CACHE SUMMARY] Unique board+bonus hashes: {len(_seen_hashes)}")
    repeated = [h for h, c in _seen_hashes.items() if c > 1]
    print(f"[CACHE SUMMARY] Hashes seen more than once: {len(repeated)}")
    if repeated:
        print(f"[CACHE SUMMARY] Example repeated hash: {repeated[0]}")
    print(f"[CACHE SUMMARY] Actual cache hits: {_actual_hits}")
    print(f"[CACHE SUMMARY] Actual cache misses: {_actual_misses}")
