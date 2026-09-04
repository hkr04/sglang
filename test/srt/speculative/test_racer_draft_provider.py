import numpy as np
import pytest

from sglang.srt.speculative.racer.draft_provider import RacerDraftProvider


class _FakeDraftSource:
    def __init__(self, draft_token_num: int, *, bad_tokens=False, bad_mask=False):
        self.draft_token_num = draft_token_num
        self.bad_tokens = bad_tokens
        self.bad_mask = bad_mask

    def batch_get(self, req_ids, batch_tokens, total_lens):
        del batch_tokens, total_lens
        bs = len(req_ids)
        num_tokens = bs * self.draft_token_num - int(self.bad_tokens)
        mask_size = bs * self.draft_token_num * self.draft_token_num - int(
            self.bad_mask
        )
        return (
            np.arange(num_tokens, dtype=np.int64),
            np.ones(mask_size, dtype=np.bool_),
        )


def test_racer_draft_provider_preserves_ngram_compatible_output():
    provider = RacerDraftProvider(_FakeDraftSource(4), draft_token_num=4)

    tokens, mask = provider.batch_get(
        req_ids=["r0", "r1"],
        batch_tokens=[[1, 2], [3, 4]],
        total_lens=[2, 2],
    )

    assert tokens.shape == (8,)
    assert mask.shape == (32,)


def test_racer_draft_provider_rejects_invalid_token_count():
    provider = RacerDraftProvider(
        _FakeDraftSource(4, bad_tokens=True), draft_token_num=4
    )

    with pytest.raises(RuntimeError, match="invalid token count"):
        provider.batch_get(["r0"], [[1]], [1])


def test_racer_draft_provider_rejects_invalid_tree_mask_size():
    provider = RacerDraftProvider(
        _FakeDraftSource(4, bad_mask=True), draft_token_num=4
    )

    with pytest.raises(RuntimeError, match="invalid tree-mask size"):
        provider.batch_get(["r0"], [[1]], [1])
