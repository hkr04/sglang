from sglang.srt.speculative.racer.automaton import _CandidateTrie


def test_candidate_trie_merge_exposes_unique_budget_holes():
    trie = _CandidateTrie()
    budget = 8

    # Two different RACER borders can recover token-identical paths.  The
    # original C++ selection budget counts both (node, start_border) entries,
    # while the candidate trie merges them into the same unique token nodes.
    trie.insert([10, 20, 30], budget)
    trie.insert([10, 20, 30], budget)

    assert trie.node_count == 3
    assert budget - trie.node_count == 5
