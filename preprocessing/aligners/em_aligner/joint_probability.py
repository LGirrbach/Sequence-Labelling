import nltk
import numpy as np

from typing import List, Union

StringSequence = Union[List[str], List[List[str]]]


class JointProbability:
    """
    Implements a wrapper around np.ndarray to store alignment scores between strings.
    The main functionality is to access entries from given string pairs, which are converted into
    row and column indices of the alignment score matrix.

    Additional functionality is the normalisation scheme, which offers either normalisation over rows
    p(y | x) or joint normalisation of the whole alignment score matrix p(y, x).
    """
    epsilon = '-'

    def __init__(self, source: List[str], target: List[List[str]], max_source_ngram: int, max_target_ngram: int,
                 normalisation_mode: str = 'conditional'):
        self.source = source
        self.target = target
        self.max_source_ngram = max_source_ngram
        self.max_target_ngram = max_target_ngram
        self.normalisation_mode = normalisation_mode

        self.probs = None

        self._make_vocab()
        self._initialise()

    @staticmethod
    def _get_ngram_vocab(seqs: StringSequence, max_ngram: int):
        """
        Generate all ngrams with length up to `max_ngram` that appear in any of the given sequences.
        """
        if len(seqs) == 0:
            return []

        all_ngrams = []
        for n in range(1, max_ngram + 1):
            # Combine ngrams from all sequences
            ngrams = set.union(*(set(nltk.ngrams(s, n)) for s in seqs))
            # Sort and convert to list
            all_ngrams.extend(list(sorted(ngrams)))

        return all_ngrams

    def _make_vocab(self):
        """
        Generate ngrams and mappings between ngrams and row/column indices.
        """
        source_ngrams = self._get_ngram_vocab(self.source, self.max_source_ngram)
        # We have to add epsilon for grapheme deletion
        source_ngrams = [(self.epsilon,)] + source_ngrams
        target_ngrams = self._get_ngram_vocab(self.target, self.max_target_ngram)
        # We have to add epsilon for phoneme deletion
        target_ngrams = [(self.epsilon,)] + target_ngrams

        # Map ngrams to indices
        self.source2idx = {s: idx for idx, s in enumerate(source_ngrams)}
        self.target2idx = {t: idx for idx, t in enumerate(target_ngrams)}

        self.num_source_ngrams = len(source_ngrams)
        self.num_target_ngrams = len(target_ngrams)

    def _initialise(self):
        """
        Initialise probability scores as uniform (either over all pairs or the conditional distributions only)
        """
        self.probs = np.ones((self.num_source_ngrams, self.num_target_ngrams))
        self.normalise(mode=self.normalisation_mode)

    def reset(self):
        """
        Set all entries of the alignment score matrix to 0
        """
        self.probs[:, :] = 0

    def __getitem__(self, key):
        # Key consists of pair (grapheme ngram, phoneme ngram)
        source, target = key
        # Convert to tuple
        source, target = tuple(source), tuple(target)
        # Get row/column indices
        source_idx = self.source2idx[source]
        target_idx = self.target2idx[target]

        # Return alignment values
        return self.probs[source_idx, target_idx]

    def __setitem__(self, key, val):
        # Key consists of pair (grapheme ngram, phoneme ngram)
        source, target = key
        source, target = tuple(source), tuple(target)
        # Get row/column indices
        source_idx = self.source2idx[source]
        target_idx = self.target2idx[target]

        # Set alignment score to given value
        self.probs[source_idx, target_idx] = val

    def normalise(self, mode='conditional'):
        if mode == 'conditional':
            self.probs /= (self.probs.sum(axis=1, keepdims=True) + 1e-7)
        elif mode == 'joint':
            self.probs /= (self.probs.sum() + 1e-7)
        elif mode == 'pmi':
            joint = self.probs / (self.probs.sum() + 1e-7)

            source_marginals = joint.sum(axis=1)
            target_marginals = joint.sum(axis=0)
            independent_joint = np.outer(source_marginals, target_marginals) + 1e-7

            pmi = np.log(joint / independent_joint)
            self.probs = pmi
        else:
            raise ValueError(f"Unknown mode: {mode}")
