"""Paged KV-cache accounting with prefix reuse.

Prompts are split into fixed-size blocks; each block is identified by a hash chained over every token before it,
so two prompts share a cached block only if they share the whole prefix up to that block (the same idea as
vLLM's automatic prefix caching / SGLang's RadixAttention).

Capacity model: ``num_blocks`` is shared between
  * cached blocks (``_lru``) - evictable only while no running sequence pins them (refcount == 0)
  * private blocks (``_reserved``) - reserved up-front by running sequences for their un-shared prompt tail
    and for the tokens they will generate
Admission (``can_admit``) guarantees ``_reserved + _pinned <= num_blocks``, so unpinned cache entries can always
be evicted to make room.
"""
from __future__ import annotations

from collections import OrderedDict
from math import ceil
from typing import Sequence


class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._lru: "OrderedDict[int, None]" = OrderedDict()  # cached block hashes, least recently used first
        self._refs: dict[int, int] = {}                      # hash -> running sequences pinning it
        self._pinned = 0                                     # cached blocks with refs > 0
        self._reserved = 0                                   # private blocks held by running sequences
        self.lookup_tokens = 0
        self.hit_tokens = 0

    # -- hashing / sizing ------------------------------------------------------------------------------
    def block_hashes(self, tokens: Sequence[int]) -> list[int]:
        bs = self.block_size
        h = 0
        out: list[int] = []
        for i in range(0, len(tokens) - bs + 1, bs):
            h = hash((h, tuple(tokens[i : i + bs])))
            out.append(h)
        return out

    def blocks_for(self, n_tokens: int) -> int:
        return ceil(n_tokens / self.block_size)

    # -- lookup / admission ----------------------------------------------------------------------------
    def match(self, hashes: Sequence[int]) -> list[int]:
        """Longest cached prefix (pure lookup: does not pin or touch anything)."""
        matched: list[int] = []
        for h in hashes:
            if h not in self._lru:
                break
            matched.append(h)
        return matched

    def can_admit(self, private_blocks: int, matched: Sequence[int]) -> bool:
        new_pins = sum(1 for h in matched if h not in self._refs)
        return self._reserved + self._pinned + private_blocks + new_pins <= self.num_blocks

    def admit(self, private_blocks: int, matched: Sequence[int], prompt_len: int) -> None:
        for h in matched:
            self._pin(h)
            self._lru.move_to_end(h)
        self._reserved += private_blocks
        self._evict_to_fit()
        self.lookup_tokens += prompt_len
        self.hit_tokens += len(matched) * self.block_size

    def commit_prefill(self, new_hashes: Sequence[int]) -> list[int]:
        """After prefill, promote this sequence's full prompt blocks from private to the shared cache.

        Returns the hashes now pinned by the sequence (to be passed to ``release``).
        """
        held: list[int] = []
        for h in new_hashes:
            if h in self._lru:
                self._lru.move_to_end(h)
            else:
                self._lru[h] = None
            self._pin(h)
            self._reserved -= 1
            held.append(h)
        return held

    def release(self, held: Sequence[int], private_blocks: int) -> None:
        for h in held:
            self._unpin(h)
        self._reserved -= private_blocks

    # -- bookkeeping -----------------------------------------------------------------------------------
    def usage(self) -> float:
        return min(1.0, (len(self._lru) + self._reserved) / self.num_blocks)

    def cached_blocks(self) -> int:
        return len(self._lru)

    def _pin(self, h: int) -> None:
        r = self._refs.get(h, 0)
        if r == 0:
            self._pinned += 1
        self._refs[h] = r + 1

    def _unpin(self, h: int) -> None:
        r = self._refs[h] - 1
        if r == 0:
            del self._refs[h]
            self._pinned -= 1
        else:
            self._refs[h] = r

    def _evict_to_fit(self) -> None:
        while len(self._lru) + self._reserved > self.num_blocks:
            for h in self._lru:  # oldest first
                if h not in self._refs:
                    del self._lru[h]
                    break
            else:
                break  # nothing evictable; cannot happen if can_admit() was honoured
