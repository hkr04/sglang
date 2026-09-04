from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class _DraftSource(Protocol):
    def batch_get(
        self,
        req_ids: Sequence[object],
        batch_tokens: Sequence[Sequence[int]],
        total_lens: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]: ...


class RacerDraftProvider:
    """Adapter boundary between RACER proposal construction and SGLang verify.

    The first integration stage intentionally delegates proposal generation to
    the existing NGRAM corpus so that the worker behavior stays unchanged. The
    RACER Aho-Corasick and logits-tree merge can replace this provider's
    ``batch_get`` implementation later without touching SGLang's tree-mask,
    TARGET_VERIFY, sampling, or KV-commit plumbing.
    """

    def __init__(self, source: _DraftSource, draft_token_num: int):
        self._source = source
        self.draft_token_num = draft_token_num

    def batch_get(
        self,
        req_ids: Sequence[object],
        batch_tokens: Sequence[Sequence[int]],
        total_lens: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        draft_tokens, tree_mask = self._source.batch_get(
            req_ids, batch_tokens, total_lens
        )
        self._validate_batch(draft_tokens, tree_mask, batch_size=len(req_ids))
        return draft_tokens, tree_mask

    def _validate_batch(
        self,
        draft_tokens: np.ndarray,
        tree_mask: np.ndarray,
        *,
        batch_size: int,
    ) -> None:
        expected_tokens = batch_size * self.draft_token_num
        expected_mask = batch_size * self.draft_token_num * self.draft_token_num

        if len(draft_tokens) != expected_tokens:
            raise RuntimeError(
                "RACER draft provider returned an invalid token count: "
                f"got {len(draft_tokens)}, expected {expected_tokens} "
                f"(batch_size={batch_size}, draft_token_num={self.draft_token_num})."
            )

        if np.size(tree_mask) != expected_mask:
            raise RuntimeError(
                "RACER draft provider returned an invalid tree-mask size: "
                f"got {np.size(tree_mask)}, expected {expected_mask} "
                f"(batch_size={batch_size}, draft_token_num={self.draft_token_num})."
            )
