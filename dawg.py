# dawg.py
# KISS first pass: compact trie with DAWG-like API.
# We can minimize nodes later; this already gives fast prefix pruning.

from typing import Dict, Iterable, List, Tuple, Optional


class DAWG:
    """
    Compact trie with the API we want:
      - DAWG.build(words) -> DAWG
      - has_prefix(str) -> bool
      - is_word(str) -> bool
      - iter_extensions(prefix) -> Iterable[(char, is_terminal)]
    Internals:
      nodes: List[{'term': bool, 'edges': Dict[str, int]}]
      node 0 is the root.
    """

    __slots__ = ("_nodes",)

    def __init__(self, nodes: List[Dict]):
        # nodes[i] = {'term': bool, 'edges': {char: child_index}}
        self._nodes = nodes

    # ---------- Public API ----------
    @classmethod
    def build(cls, words: Iterable[str]) -> "DAWG":
        """
        Build a trie from the given words. Words are uppercased and filtered to A-Z.
        """
        nodes: List[Dict[str, object]] = [{"term": False, "edges": {}}]  # root at 0

        def add_word(w: str):
            cur = 0
            edges0 = nodes[cur]["edges"]
            for ch in w:
                nxt = edges0.get(ch)
                if nxt is None:
                    nodes.append({"term": False, "edges": {}})
                    nxt = len(nodes) - 1
                    edges0[ch] = nxt
                cur = nxt
                edges0 = nodes[cur]["edges"]
            nodes[cur]["term"] = True

        for w in words:
            if not w:
                continue
            # Normalize: uppercase and keep A–Z only (your pipeline already uppercases)
            ww = "".join(ch for ch in w.upper() if "A" <= ch <= "Z")
            if len(ww) >= 1:
                add_word(ww)

        return cls(nodes)

    def has_prefix(self, s: str) -> bool:
        """True if s is a path from the root (empty string is always a prefix)."""
        idx = self._walk(s)
        return idx is not None

    def is_word(self, s: str) -> bool:
        """True if s is in the trie as a terminal word."""
        idx = self._walk(s)
        return (idx is not None) and bool(self._nodes[idx]["term"])

    def iter_extensions(self, prefix: str) -> Iterable[Tuple[str, bool]]:
        """
        Yield (next_char, is_terminal_after_appending_char) for all single-letter
        continuations of 'prefix'. If prefix isn’t present, yields nothing.
        """
        idx = self._walk(prefix)
        if idx is None:
            return
        edges: Dict[str, int] = self._nodes[idx]["edges"]
        for ch, child in edges.items():
            yield ch, bool(self._nodes[child]["term"])

    # ---------- Helpers ----------
    def _walk(self, s: str) -> Optional[int]:
        """Return node index after consuming s, or None if no such path."""
        idx = 0
        nodes = self._nodes
        for ch in s:
            edges: Dict[str, int] = nodes[idx]["edges"]  # type: ignore[assignment]
            nxt = edges.get(ch)
            if nxt is None:
                return None
            idx = nxt
        return idx