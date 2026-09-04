from sglang.srt.speculative.ngram_worker import NGRAMWorker
from sglang.srt.speculative.racer.draft_provider import RacerDraftProvider


class RACERWorker(NGRAMWorker):
    """Initial RACER worker shell reusing SGLang's NGRAM verify pipeline.

    At this stage proposal generation is still delegated to the wrapped
    ``NgramCorpus``. The only behavioral change is the insertion of
    ``RacerDraftProvider`` as the boundary where RACER's AC/logits-tree proposal
    generation will be added in later commits.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ngram_corpus = RacerDraftProvider(
            self.ngram_corpus,
            draft_token_num=self.draft_token_num,
        )
