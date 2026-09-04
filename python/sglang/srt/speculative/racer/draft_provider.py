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
    the existing NGRAM corpus so that worker behavior stays unchanged. The
    provider proxies the rest of the corpus API, which lets ``RACERWorker``
    reuse SGLang's existing NGRAM lifecycle, tree-mask preparation,
    TARGET_VERIFY, sampling, and KV-commit plumbing without modifying
    ``NGRAMWorker`` itself.

    RACER's Aho-Corasick and logits-tree merge can later replace ``batch_get``
    while keeping the surrounding execution path intact.
    """

    def __init__(self, source: _DraftSource, draft_token_num: int):
        self._source = source
        self.draft_token_num = draft_token_num

    def __getattr__(self, name: str):
        """Forward non-drafting corpus operations to the wrapped source."""
        return getattr(self._source, name)

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
