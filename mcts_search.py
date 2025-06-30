# --- mcts_search.py ---

import random
from collections import Counter
import mcts

from utils import N
from board import is_letter, place_word, board_valid, compute_board_score
from search import can_play_word_on_board, is_valid_placement


class ScrapleState:
    def __init__(self, board, rack_count, words, wordset, original_bonus, depth=0, max_depth=5, moves=None):
        self.board = [row[:] for row in board]
        self.rack_count = rack_count.copy()
        self.words = words
        self.wordset = wordset
        self.original_bonus = original_bonus
        self.depth = depth
        self.max_depth = max_depth
        self.moves = moves or []

    # Generate all legal placements for the current state
    def getPossibleActions(self):
        if self.depth >= self.max_depth:
            return []
        actions = []
        for w in self.words:
            L = len(w)
            for r in range(N):
                for c in range(N - L + 1):
                    if not is_valid_placement(w, self.board, self.rack_count, self.wordset, r, c, 'H'):
                        continue
                    can_play, _ = can_play_word_on_board(w, r, c, 'H', self.board, self.rack_count)
                    if can_play:
                        actions.append(('H', w, r, c))
            for r in range(N - L + 1):
                for c in range(N):
                    if not is_valid_placement(w, self.board, self.rack_count, self.wordset, r, c, 'V'):
                        continue
                    can_play, _ = can_play_word_on_board(w, r, c, 'V', self.board, self.rack_count)
                    if can_play:
                        actions.append(('V', w, r, c))
        return actions

    def takeAction(self, action):
        direction, word, r0, c0 = action
        board_copy = [row[:] for row in self.board]
        rack_copy = self.rack_count.copy()
        _, rack_after = can_play_word_on_board(word, r0, c0, direction, board_copy, rack_copy)
        place_word(board_copy, word, r0, c0, direction)
        new_words = [w for w in self.words if w != word]
        moves = self.moves + [(word, direction, r0, c0)]
        return ScrapleState(board_copy, rack_after, new_words, self.wordset, self.original_bonus,
                            depth=self.depth + 1, max_depth=self.max_depth, moves=moves)

    def isTerminal(self):
        if self.depth >= self.max_depth:
            return True
        return len(self.getPossibleActions()) == 0

    def getReward(self):
        return compute_board_score(self.board, self.original_bonus)


def rollout_policy(state: ScrapleState):
    current = state
    while not current.isTerminal():
        actions = current.getPossibleActions()
        if not actions:
            break
        action = random.choice(actions)
        current = current.takeAction(action)
    return current.getReward()


def run_mcts_search(board, rack, words, wordset, original_bonus, iterations=1000, max_depth=5):
    rack_count = Counter(rack)
    root_state = ScrapleState(board, rack_count, words, wordset, original_bonus, max_depth=max_depth)
    algo = mcts.mcts(iterationLimit=iterations, rolloutPolicy=rollout_policy)
    algo.search(root_state)

    node = algo.root
    actions = []
    while node.children:
        best_child = algo.getBestChild(node, 0)
        for act, child in node.children.items():
            if child is best_child:
                actions.append(act)
                node = child
                break
        else:
            break

    final_state = node.state
    return final_state.getReward(), final_state.board, final_state.moves

