from board import compute_board_score
import hashlib
import utils


from collections import OrderedDict

_seen_hashes = {}
_actual_hits = 0
_actual_misses = 0
CACHE_DISABLED = False

# LRU cache using OrderedDict
MAX_CACHE_SIZE = 50000
class LRUCache(OrderedDict):
    def __init__(self, maxsize=MAX_CACHE_SIZE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxsize = maxsize
    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value
    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


def board_to_tuple(board):
    return tuple(tuple(row) for row in board)


def board_hash(board_tuple, bonus_tuple):
    s = str(board_tuple) + str(bonus_tuple)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def cached_board_score(board_tuple, bonus_tuple, cache=None):
    """Compute or retrieve a cached board score.

    Uses a compact hash key to reduce key size and enables a simple size cap
    to avoid unbounded memory growth. When CACHE_DISABLED is True, always
    recomputes without touching cache counters.
    """
    global _actual_hits, _actual_misses

    if CACHE_DISABLED:
        return compute_board_score(board_tuple, bonus_tuple)

    # Verbose hash tracking for diagnostics
    if utils.VERBOSE:
        h = board_hash(board_tuple, bonus_tuple)
        count = _seen_hashes.get(h, 0)
        _seen_hashes[h] = count + 1


    if cache is None:
        cache = getattr(cached_board_score, "_cache", None)
        if cache is None:
            cache = LRUCache(MAX_CACHE_SIZE)
            cached_board_score._cache = cache

    key = board_hash(board_tuple, bonus_tuple)
    if key in cache:
        _actual_hits += 1
        return cache[key]

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