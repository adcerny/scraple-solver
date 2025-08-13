import time
import math
import random
from collections import Counter
from typing import List, Optional, Tuple

from utils import log_with_time, Direction
from board import place_word
from score_cache import cached_board_score, board_to_tuple
from search import find_best, prune_words, full_beam_search


class Node:
    def __init__(
        self,
        board,
        rack: Counter,
        remaining_words: List[str],
        depth: int,
        parent: Optional["Node"] = None,
        move=None,
    ):
        self.board = board
        self.rack = rack
        self.remaining_words = remaining_words
        self.depth = depth
        self.parent = parent
        self.move = move  # (score, word, direction, row, col)
        self.visits = 0
        self.value = 0.0
        self.children: List[Node] = []
        self.untried_moves: Optional[List[Tuple]] = None


class MCTS:
    def __init__(
        self,
        board,
        rack,
        words,
        wordset,
        prefixset,
        original_bonus,
        top_k: int = 10,
        epsilon: float = 0.2,
        uct_c: float = 1.2,
        max_depth: int = 20,
        use_beam_rollout: bool = False,
    ):
        self.top_k = top_k
        self.epsilon = epsilon
        self.uct_c = uct_c
        self.max_depth = max_depth
        self.use_beam_rollout = use_beam_rollout
        self.wordset = wordset
        self.prefixset = prefixset
        self.original_bonus = original_bonus
        self.original_bonus_tuple = board_to_tuple(original_bonus)

        rack_counter = Counter(rack)
        pruned_words = prune_words(words, rack_counter, board)
        root_board = [row[:] for row in board]
        self.root = Node(root_board, rack_counter, pruned_words, depth=0)

    # Helper to get candidate moves for a node
    def _ensure_moves(self, node: Node):
        if node.untried_moves is not None:
            return
        if node.depth >= self.max_depth or not node.rack:
            node.untried_moves = []
            return
        raw_moves = find_best(
            node.board,
            node.rack,
            node.remaining_words,
            self.wordset,
            prefixset=self.prefixset,
            touch=None,
            original_bonus=self.original_bonus,
            top_k=self.top_k * 5,
        )
        if not raw_moves:
            node.untried_moves = []
            return

        best_by_word = {}
        for sc, w, d, r, c in raw_moves:
            existing = best_by_word.get(w)
            if existing is None or sc > existing[0]:
                best_by_word[w] = (sc, w, d, r, c)
        deduped = sorted(best_by_word.values(), key=lambda m: m[0], reverse=True)[: self.top_k]
        node.untried_moves = list(deduped)

    def _uct_select(self, node: Node) -> Node:
        log_parent = math.log(node.visits)
        return max(
            node.children,
            key=lambda n: n.value / n.visits + self.uct_c * math.sqrt(log_parent / n.visits),
        )

    def _apply_move(self, board, rack: Counter, remaining_words: List[str], move):
        score, word, direction, row, col = move
        new_board = [r[:] for r in board]
        place_word(new_board, word, row, col, direction)
        new_rack = rack.copy()
        for i, ch in enumerate(word):
            r = row + (i if direction == Direction.DOWN else 0)
            c = col + (i if direction == Direction.ACROSS else 0)
            if len(board[r][c]) != 1:
                new_rack[ch] -= 1
        new_remaining = remaining_words.copy()
        try:
            while True:
                new_remaining.remove(word)
        except ValueError:
            pass
        return new_board, new_rack, new_remaining

    def _select(self) -> Tuple[Node, List[Tuple]]:
        node = self.root
        path = []
        self._ensure_moves(node)
        while node.children and not node.untried_moves:
            node = self._uct_select(node)
            path.append(node.move)
            self._ensure_moves(node)
        return node, path

    def _expand(self, node: Node) -> Node:
        if not node.untried_moves:
            return node
        move = node.untried_moves.pop(0)
        board, rack, rem = self._apply_move(node.board, node.rack, node.remaining_words, move)
        child = Node(board, rack, rem, node.depth + 1, parent=node, move=move)
        node.children.append(child)
        self._ensure_moves(child)
        return child

    def _rollout(self, node: Node):
        if self.use_beam_rollout:
            placed = {
                (r, c)
                for r in range(len(node.board))
                for c in range(len(node.board[r]))
                if len(node.board[r][c]) == 1
            }
            max_moves = self.max_depth - node.depth
            score, board_end, moves = full_beam_search(
                node.board,
                node.rack,
                node.remaining_words,
                self.wordset,
                self.prefixset,
                placed,
                self.original_bonus,
                beam_width=self.top_k,
                max_moves=max_moves,
            )
            return score, moves, board_end

        board = [r[:] for r in node.board]
        rack = node.rack.copy()
        remaining = node.remaining_words.copy()
        depth = node.depth
        moves = []
        while depth < self.max_depth and rack:
            candidates = find_best(
                board,
                rack,
                remaining,
                self.wordset,
                prefixset=self.prefixset,
                touch=None,
                original_bonus=self.original_bonus,
                top_k=self.top_k,
            )
            if not candidates:
                break
            if random.random() < self.epsilon:
                idx = random.randrange(min(len(candidates), 12))
                move = candidates[idx]
            else:
                move = candidates[0]
            board, rack, remaining = self._apply_move(board, rack, remaining, move)
            moves.append(move)
            depth += 1
        final_score = cached_board_score(board_to_tuple(board), self.original_bonus_tuple)
        return final_score, moves, board

    def _backpropagate(self, node: Node, reward: float):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def search(self, iters: Optional[int] = None, seconds: Optional[float] = None):
        start = time.time()
        iterations = 0
        best_score = float("-inf")
        best_line = []
        best_board = None
        best_found_iter = None
        best_found_time = None

        while True:
            if iters is not None and iterations >= iters:
                break
            if seconds is not None and time.time() - start >= seconds:
                break
            node, path = self._select()
            node = self._expand(node)
            reward, rollout_moves, final_board = self._rollout(node)
            self._backpropagate(node, reward)
            iterations += 1

            full_line = path + rollout_moves
            if reward > best_score:
                best_score = reward
                best_line = full_line
                best_board = final_board
                best_found_iter = iterations
                best_found_time = time.time() - start

        elapsed = time.time() - start
        log_with_time(f"MCTS completed {iterations} iterations in {elapsed:.2f}s")
        if best_line:
            log_with_time(
                f"Best score {best_score} first found at iteration {best_found_iter} after {best_found_time:.2f}s",
            )
            log_with_time("Best line:")
            for sc, w, d, r0, c0 in best_line:
                log_with_time(f"  {w} at {r0},{c0},{d.value} -> {sc}")
        return best_score, best_board, best_line
