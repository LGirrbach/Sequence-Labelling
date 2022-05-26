import nltk
import torch

from typing import List
from typing import Union
from typing import Tuple
from collections import Counter


class SequenceLabellingVocabulary:
    PAD = "<PAD>"
    UNK = "<UNK>"
    SPECIALS = [PAD, UNK]

    def __init__(self, x: List[Union[str, List[str]]], y: List[Union[str, List[str]]]):
        # Collect vocabulary
        self.source_vocab = self.SPECIALS + list(sorted(set.union(*(set(s) for s in x))))
        self.target_vocab = self.SPECIALS + list(sorted(set.union(*(set(t) for t in y))))

        # Build mappings
        self.source2idx = {word: idx for idx, word in enumerate(self.source_vocab)}
        self.target2idx = {word: idx for idx, word in enumerate(self.target_vocab)}
        self.idx2source = {idx: word for idx, word in enumerate(self.source_vocab)}
        self.idx2target = {idx: word for idx, word in enumerate(self.target_vocab)}

        # Save UNK indices
        self.source_unk_idx = self.source2idx[self.UNK]
        self.target_unk_idx = self.target2idx[self.UNK]

    def num_target(self):
        return len(self.target_vocab)

    def num_source(self):
        return len(self.source_vocab)

    def index_source(self, source: Union[str, List[str]]):
        return torch.LongTensor([self.source2idx.get(s, self.source_unk_idx) for s in source])

    def index_target(self, target: Union[str, List[str]]):
        return torch.LongTensor([self.target2idx.get(t, self.source_unk_idx) for t in target])

    def decode_source(self, source_idx: List[int]):
        return [self.idx2source[(int(idx))] for idx in source_idx]

    def decode_target(self, target_idx: List[int]):
        return [self.idx2target[(int(idx))] for idx in target_idx]

    def source_is_special(self, item: Union[int, str]):
        if isinstance(item, int):
            return self.idx2source[item] in self.SPECIALS
        elif isinstance(item, str):
            return item in self.SPECIALS
        else:
            raise TypeError(f"Item must be str or int, not {type(item)}")

    def target_is_special(self, item: Union[int, str]):
        if isinstance(item, int):
            return self.idx2target[item] in self.SPECIALS
        elif isinstance(item, str):
            return item in self.SPECIALS
        else:
            raise TypeError(f"Item must be str or int, not {type(item)}")


class GramCTCVocabulary(SequenceLabellingVocabulary):
    PAD = (SequenceLabellingVocabulary.PAD,)
    UNK = (SequenceLabellingVocabulary.UNK,)
    BLANK = ("<BLANK>",)
    SPECIALS = [PAD, UNK, BLANK]

    def __init__(self, x: List[Union[str, List[str]]], y: List[Union[str, List[str]]],
                 allowed_ngrams: List[Tuple[str]] = None, tau: int = 3, ngram_count_threshold: float = 3):
        super(GramCTCVocabulary, self).__init__(x, y)

        self.tau = tau
        self.ngram_count_threshold = ngram_count_threshold

        if allowed_ngrams is not None:
            assert all(isinstance(ngram, tuple) and all(isinstance(c, str) for c in ngram) for ngram in allowed_ngrams)
            allowed_ngrams = self.SPECIALS + [tuple(ngram) for ngram in allowed_ngrams]

        elif tau >= 0:
            all_ngrams = []
            for target in y:
                for n in range(1, tau + 1):
                    for ngram in nltk.ngrams(target, n):
                        all_ngrams.append(ngram)

            ngram_counts = Counter(all_ngrams)
            allowed_ngrams = [ngram for ngram, count in ngram_counts.items() if count >= self.ngram_count_threshold]
            allowed_ngrams = list(sorted(allowed_ngrams))
            allowed_ngrams = self.SPECIALS + allowed_ngrams

        else:
            raise RuntimeError("Must provide either ngram whitelist or valid max. ngram length")

        self.target_vocab = allowed_ngrams
        self.target2idx = {ngram: idx for idx, ngram in enumerate(allowed_ngrams)}
        self.idx2target = {idx: ngram for ngram, idx in self.target2idx.items()}

    def index_target(self, target: Union[str, List[str]]):
        return list(target)


class ScatterCTCVocabulary(SequenceLabellingVocabulary):
    PAD = "<PAD>"
    UNK = "<UNK>"
    BLANK = "<BLANK>"
    SPECIALS = [PAD, UNK, BLANK]

    def __init__(self, x: List[Union[str, List[str]]], y: List[Union[str, List[str]]], tau: int = 3,
                 repeat: bool = True):
        super(ScatterCTCVocabulary, self).__init__(x=x, y=y)

        self.tau = tau
        self.repeat = repeat
        self.blank_idx = self.SPECIALS.index(self.BLANK)

    def index_source(self, source: Union[str, List[str]]):
        source = list(source)
        scattered_source = []
        for s in source:
            # Add s `self.tau` times
            for k in range(self.tau):
                scattered_source.append(s if self.repeat or k == 0 else self.BLANK)

        return torch.LongTensor([self.source2idx.get(s, self.source_unk_idx) for s in scattered_source])
