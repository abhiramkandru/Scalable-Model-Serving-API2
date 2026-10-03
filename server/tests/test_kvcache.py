from app.kvcache import KVCacheManager


def run_seq(kv: KVCacheManager, tokens: list[int], max_tokens: int, finish: bool = True):
    """Mimic the scheduler: match -> admit -> prefill commit -> (optionally) release."""
    hashes = kv.block_hashes(tokens)
    matched = kv.match(hashes)
    private = kv.blocks_for(len(tokens) + max_tokens) - len(matched)
    assert kv.can_admit(private, matched)
    kv.admit(private, matched, len(tokens))
    new = hashes[len(matched):]
    held = list(matched) + kv.commit_prefill(new)
    private -= len(new)
    if finish:
        kv.release(held, private)
        return len(matched)
    return held, private


def test_match_requires_the_whole_shared_prefix():
    kv = KVCacheManager(num_blocks=64, block_size=4)
    a = list(range(16))
    assert run_seq(kv, a, 4) == 0                      # cold cache
    b = a[:8] + [900 + i for i in range(8)]            # shares the first two blocks only
    assert len(kv.match(kv.block_hashes(b))) == 2
    c = [999] + a[1:]                                  # differs in token 0 => nothing may match (chained hash)
    assert kv.match(kv.block_hashes(c)) == []


def test_repeat_prompt_hits_cache_and_counts_tokens():
    kv = KVCacheManager(num_blocks=64, block_size=4)
    a = list(range(18))                                # 4 full blocks + 2 tail tokens
    run_seq(kv, a, 4)
    assert run_seq(kv, a, 4) == 4
    assert kv.hit_tokens == 16 and kv.lookup_tokens == 36


def test_accounting_returns_to_zero():
    kv = KVCacheManager(num_blocks=64, block_size=4)
    for i in range(10):
        run_seq(kv, list(range(12)) + [i, i + 1, i + 2, i + 3], 8)
    assert kv._reserved == 0 and kv._pinned == 0 and not kv._refs
    assert kv.usage() <= 1.0


def test_pinned_blocks_are_never_evicted_and_block_admission():
    kv = KVCacheManager(num_blocks=10, block_size=4)
    held, private = run_seq(kv, list(range(16)), 4, finish=False)   # 4 prompt blocks pinned + 1 private
    other = [500 + i for i in range(16)]
    h = kv.block_hashes(other)
    need = kv.blocks_for(len(other) + 4)
    assert not kv.can_admit(need + 1, kv.match(h))                 # 5 in use + 6 > 10: must wait
    kv.release(held, private)
    assert kv.can_admit(need + 1, [])                              # freed: unpinned blocks are evictable


def test_lru_eviction_keeps_capacity_and_drops_oldest():
    kv = KVCacheManager(num_blocks=8, block_size=4)
    prompts = [[1000 * k + i for i in range(12)] for k in range(6)]  # 3 blocks each
    for p in prompts:
        run_seq(kv, p, 1)
        assert kv.cached_blocks() <= 8
    assert len(kv.match(kv.block_hashes(prompts[-1]))) == 3         # newest survives
    assert kv.match(kv.block_hashes(prompts[0])) == []              # oldest was evicted


def test_concurrent_identical_prefill_shares_blocks():
    kv = KVCacheManager(num_blocks=32, block_size=4)
    a = list(range(16))
    h1, p1 = run_seq(kv, a, 4, finish=False)
    h2, p2 = run_seq(kv, a, 4, finish=False)                       # second sees the first's committed blocks
    assert kv.cached_blocks() == 4 and kv.hit_tokens == 16         # all four blocks shared, not duplicated
    kv.release(h1, p1)
    kv.release(h2, p2)
    assert kv._reserved == 0 and kv._pinned == 0
