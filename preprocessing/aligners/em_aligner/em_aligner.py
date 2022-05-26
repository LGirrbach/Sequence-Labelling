import os
import pickle
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from logger import logger as logging
from preprocessing.aligners.em_aligner.joint_probability import JointProbability

from typing import List


class EMAligner:
    """
    Implements the Expectation-Maximisation many-to-many aligner from
    Jiampojamarn et al. (2007): https://aclanthology.org/N07-1047/

    The following modifications have been implemented:
      * Phonemes cannot be deleted (every phoneme must be aligned to a grapheme)
      * Optionally only allow 1-to-many and many-to-1 mappings
    """

    def __init__(self, max_source_ngram: int, max_target_ngram: int, normalisation_mode: str = 'conditional',
                 many2many: bool = True, allow_delete_graphemes: bool = True, allow_delete_phonemes: bool = False,
                 epochs: int = 1, **params):

        self.max_source_ngram = max_source_ngram
        self.max_target_ngram = max_target_ngram
        self.normalisation_mode = normalisation_mode
        self.many2many = many2many
        self.allow_delete_graphemes = allow_delete_graphemes
        self.allow_delete_phonemes = allow_delete_phonemes
        self.epochs = epochs

        self.epsilon = JointProbability.epsilon
        self.gamma = None

        self.is_fit = False

    def save(self, path):
        """
        Save model at given location.

        The model is saved in 2 files: `parameters.json` contains model parameter values and `probabilities.npy`
        contains the alignment probability matrix.
        """
        if not self.is_fit:
            raise RuntimeError("Cannot save untrained model")

        logging.info(f"Saving model to {path}")

        # Create path if necessary
        if not os.path.exists(path):
            os.makedirs(path)

        # Save parameters of aligner
        aligner_parameters = [
            self.max_source_ngram, self.max_target_ngram, self.normalisation_mode, self.many2many,
            self.allow_delete_graphemes, self.allow_delete_phonemes, self.epochs
        ]

        # Save parameters of alignment probability matrix
        joint_probability_parameters = [
            self.gamma.source, self.gamma.target, self.gamma.max_source_ngram, self.gamma.max_target_ngram,
            self.gamma.normalisation_mode, self.gamma.source2idx, self.gamma.target2idx, self.gamma.num_source_ngrams,
            self.gamma.num_target_ngrams
        ]

        with open(os.path.join(path, "parameters.pickle"), "wb") as sf:
            pickle.dump([aligner_parameters, joint_probability_parameters], sf)

        # Use special numpy functionality to save alignment probability matrix
        np.save(os.path.join(path, "probabilities"), self.gamma.probs)
        logging.info(f"Successfully saved model to {path}")

    @staticmethod
    def load(path):
        """
        Load model from given location.

        The model is saved in 2 files: `parameters.json` contains model parameter values and `probabilities.npy`
        contains the alignment probability matrix.
        """
        # Load parameters from file
        with open(os.path.join(path, "parameters.pickle"), "rb") as sf:
            aligner_parameters, joint_probability_parameters = pickle.load(sf)

        # Create instance of EMAligner
        max_source_ngram = aligner_parameters[0]
        max_target_ngram = aligner_parameters[1]
        normalisation_mode = aligner_parameters[2]
        many2many = aligner_parameters[3]
        allow_delete_graphemes = aligner_parameters[4]
        allow_delete_phonemes = aligner_parameters[5]
        epochs = aligner_parameters[6]

        aligner = EMAligner(max_source_ngram=max_source_ngram, max_target_ngram=max_target_ngram,
                            normalisation_mode=normalisation_mode, many2many=many2many,
                            allow_delete_graphemes=allow_delete_graphemes, allow_delete_phonemes=allow_delete_phonemes,
                            epochs=epochs)
        aligner.is_fit = True

        # Create alignment probabilities
        gamma = JointProbability([], [], max_source_ngram, max_target_ngram)
        gamma.source = joint_probability_parameters[0]
        gamma.target = joint_probability_parameters[1]
        gamma.max_source_ngram = joint_probability_parameters[2]
        gamma.max_target_ngram = joint_probability_parameters[3]
        gamma.normalisation_mode = joint_probability_parameters[4]
        gamma.source2idx = joint_probability_parameters[5]
        gamma.target2idx = joint_probability_parameters[6]
        gamma.num_source_ngrams = joint_probability_parameters[7]
        gamma.num_target_ngrams = joint_probability_parameters[8]
        gamma.probs = np.load(os.path.join(path, "probabilities.npy"))

        aligner.gamma = gamma

        return aligner

    def _forward(self, source: str, target: List[str], delta: JointProbability):
        """
        Forward pass of forward-backward algorithm.
        Implements Algorithm 3 from Jiampojamarn et al. (2007).

        :param source: Sequence of graphemes
        :param target: Sequence of phonemes
        :param delta: Alignment probabilities
        :return: Forward scores `alpha`
        """
        source_length, target_length = len(source), len(target)

        # Initialise forward scores to 0
        alpha = np.zeros(shape=(source_length + 1, target_length + 1))
        # Initialise upper left corner to 1
        alpha[0, 0] = 1

        # Calculate forward scores for each grapheme-phoneme pair
        # Iterate over all rows
        for t in range(0, source_length + 1):
            # Iterate over all columns
            for v in range(0, target_length + 1):
                # Add scores for grapheme deletion
                if t > 0 and self.allow_delete_graphemes:
                    for i in range(1, min(t, self.max_source_ngram) + 1):
                        s_ngram = source[t - i:t]
                        alpha[t, v] += delta[s_ngram, self.epsilon] * alpha[t - i, v]

                # Add scores for phoneme deletion
                if v > 0 and self.allow_delete_phonemes:
                    for j in range(1, min(v, self.max_target_ngram) + 1):
                        t_ngram = target[v - j:v]
                        alpha[t, v] += delta[self.epsilon, t_ngram] * alpha[t, v - j]

                # Add scores for grapheme-phoneme ngram alignment
                if v > 0 and t > 0:
                    for i in range(1, min(t, self.max_source_ngram) + 1):
                        for j in range(1, min(v, self.max_target_ngram) + 1):
                            # Check if matching ngrams is allowed
                            if i > 1 and j > 1 and not self.many2many:
                                continue

                            s_ngram = source[t - i:t]
                            t_ngram = target[v - j:v]
                            alpha[t, v] += delta[s_ngram, t_ngram] * alpha[t - i, v - j]

        return alpha

    def _backward(self, source: str, target: List[str], delta: JointProbability):
        """
        Backward pass of forward-backward algorithm.
        Analog to Algorithm 3 in Jiampojamarn et al. (2007) and the forward pass.

        :param source: Sequence of graphemes
        :param target: Sequence of phonemes
        :param delta: Alignment probabilities
        :return: Backward scores `beta`
        """
        source_length, target_length = len(source), len(target)

        # Initialise backward scores to 0
        beta = np.zeros(shape=(source_length + 1, target_length + 1))
        # Initialise lower right corner to 1
        beta[source_length, target_length] = 1

        # Calculate backward scores for each grapheme-phoneme pair
        # Iterate over all rows (from bottom to top)
        for t in reversed(range(0, source_length + 1)):
            # Iterate over all columns (from left to right
            for v in reversed(range(0, target_length + 1)):
                # Add scores for grapheme deletion
                if t < source_length and self.allow_delete_graphemes:
                    for i in range(1, min(source_length - t, self.max_source_ngram) + 1):
                        s_ngram = source[t:t + i]
                        beta[t, v] += delta[s_ngram, self.epsilon] * beta[t + i, v]

                # Add scores for phoneme deletion
                if v < target_length and self.allow_delete_phonemes:
                    for j in range(1, min(target_length - v, self.max_target_ngram) + 1):
                        t_ngram = target[v:v + j]
                        beta[t, v] += delta[self.epsilon, t_ngram] * beta[t, v + j]

                # Add scores for grapheme-phoneme ngram alignment
                if v < target_length and t < source_length:
                    for i in range(1, min(source_length - t, self.max_source_ngram) + 1):
                        for j in range(1, min(target_length - v, self.max_target_ngram) + 1):
                            # Check if matching ngrams is allowed
                            if i > 1 and j > 1 and not self.many2many:
                                continue

                            s_ngram = source[t:t + i]
                            t_ngram = target[v:v + j]
                            beta[t, v] += delta[s_ngram, t_ngram] * beta[t + i, v + j]

        return beta

    def _expectation(self, source: str, target: List[str], gamma: JointProbability, delta: JointProbability):
        """
        Calculate expectation scores of forward-backward algorithm.
        Implements Algorithm 2 in Jiampojamarn et al. (2007).

        :param source: Sequence of graphemes
        :param target: Sequence of phonemes
        :param gamma: Matching probabilities to update
        :param delta: Matching probabilities (remains unchanged)
        :return: Expectation scores `gamma`
        """
        source_length, target_length = len(source), len(target)

        # Calculate forward and backward scores
        alpha = self._forward(source, target, delta)
        beta = self._backward(source, target, delta)

        # If alignment has 0 probability, don't perform any updates (avoid division by 0)
        if alpha[source_length, target_length] == 0.0:
            return gamma

        # Define normaliser for scores
        normaliser = alpha[source_length, target_length]

        # Calculate expectation scores:
        # Iterate over rows (from top to bottom)
        for t in range(0, source_length + 1):
            for v in range(0, target_length + 1):
                # Calculate scores for grapheme deletion
                if t > 0 and self.allow_delete_graphemes:
                    for i in range(1, min(t, self.max_source_ngram) + 1):
                        s_ngram = source[t - i:t]
                        score = (alpha[t - i, v] * delta[s_ngram, self.epsilon] * beta[t, v]) / normaliser
                        gamma[s_ngram, self.epsilon] += score

                # Calculate scores for phoneme deletion
                if v > 0 and self.allow_delete_phonemes:
                    for j in range(1, min(v, self.max_target_ngram) + 1):
                        t_ngram = target[v - j:v]
                        score = (alpha[t, v - j] * delta[self.epsilon, t_ngram] * beta[t, v]) / normaliser
                        gamma[self.epsilon, t_ngram] += score

                # Calculate scores for grapheme-phoneme ngram alignment
                if v > 0 and t > 0:
                    for i in range(1, min(t, self.max_source_ngram) + 1):
                        for j in range(1, min(v, self.max_target_ngram) + 1):
                            # Check if matching ngrams is allowed
                            if i > 1 and j > 1 and not self.many2many:
                                continue

                            s_ngram = source[t - i:t]
                            t_ngram = target[v - j:v]
                            score = (alpha[t - i, v - j] * delta[s_ngram, t_ngram] * beta[t, v]) / normaliser
                            gamma[s_ngram, t_ngram] += score

        return gamma

    def fit(self, source: List[str], target: List[List[str]]):
        """
        Perform several iterations of EM algorithm to learn ngram alignment and deletion probabilities for
        given grapheme-phoneme pairs.
        Implements Algorithm 1 in Jiampojamarn et al. (2007).

        :param source: Grapheme sequences
        :param target: Phoneme sequences
        :return: None
        """
        # Initialise ngram alignment probabilities
        logging.info("Initialise parameters")
        gamma = JointProbability(source, target, self.max_source_ngram, self.max_target_ngram,
                                 normalisation_mode=self.normalisation_mode)

        # Run EM for `epochs` many iterations
        logging.info("Start training")
        pbar = tqdm(total=len(source) * self.epochs, desc="Training progress")
        for epoch in range(self.epochs):
            # Create copy of ngram alignment probabilities
            delta = deepcopy(gamma)
            # Maximisation step
            delta.normalise(mode=self.normalisation_mode)
            # Set all alignment probabilities to 0 (only in `gamma`)
            gamma.reset()

            # Add expectation scores for each pair of graphemes/phonemes
            for s, t in zip(source, target):
                gamma = self._expectation(s, t, gamma, delta)
                pbar.update(1)

        pbar.close()
        logging.info("Finished training")

        # Maximisation step
        gamma.normalise(mode=self.normalisation_mode)
        # Save data
        self.gamma = gamma
        self.is_fit = True

    def _align_instance(self, source: List[str], target: List[str]):
        """
        Align graphemes given in `source` to phonemes given in `target` according to learned ngram alignment
        probabilities stored in `self.gamma`.

        Returns tuple of lists: The first lists contains the aligned grapheme chunks and the second list contains
        the aligned phoneme chunks. Both lists have the same length. Each chunk is represented by a list of strings.

        The implementation is equivalent to the forward pass of the forward backward algorithm, except that we
        take the maximum probability path instead of the expected probability.

        :param source: Sequence of graphemes
        :param target: Sequence of phonemes
        :return: Aligned grapheme and phoneme chunks
        """
        source_length, target_length = len(source), len(target)
        delta = self.gamma

        # Initialise log-probs as -infinity (prob = 0)
        alpha = np.full((source_length + 1, target_length + 1), -np.inf)
        # Initialise log prob of upper left corner to 0 (prob = 1)
        alpha[0, 0] = 0
        # Initialise traceback
        traceback = [[(0, 0) for _ in range(target_length + 1)] for _ in range(source_length + 1)]

        # Calculate maximum probability alignment path
        # Iterate over rows (from top to bottom)
        for t in range(0, source_length + 1):
            # Iterate over columns (from left to right)
            for v in range(0, target_length + 1):
                # Check grapheme deletion score
                if t > 0 and self.allow_delete_graphemes:
                    for i in range(1, min(t, self.max_source_ngram) + 1):
                        s_ngram = source[t - i:t]
                        # Clamp alignment probability for numerical stability
                        # And avoid problems if all paths have 0 probability
                        score = alpha[t - i, v] + np.log(max(delta[s_ngram, self.epsilon], 1e-7))

                        if score >= alpha[t, v]:
                            alpha[t, v] = score
                            traceback[t][v] = (i, 0)

                # Check phoneme deletion score
                if v > 0 and self.allow_delete_phonemes:
                    for j in range(1, min(v, self.max_target_ngram) + 1):
                        t_ngram = target[v - j:v]
                        # Clamp alignment probability for numerical stability
                        # And avoid problems if all paths have 0 probability
                        score = alpha[t, v - j] + np.log(max(delta[self.epsilon, t_ngram], 1e-7))

                        if score >= alpha[t, v]:
                            alpha[t, v] = score
                            traceback[t][v] = (0, j)

                # Check grapheme-phoneme alignment scores
                if v > 0 and t > 0:
                    for i in range(1, min(t, self.max_source_ngram) + 1):
                        for j in range(1, min(v, self.max_target_ngram) + 1):
                            s_ngram = source[t - i:t]
                            t_ngram = target[v - j:v]
                            # Clamp alignment probability for numerical stability
                            # And avoid problems if all paths have 0 probability
                            score = alpha[t - i, v - j] + np.log(max(delta[s_ngram, t_ngram], 1e-7))

                            if score >= alpha[t, v]:
                                alpha[t, v] = score
                                traceback[t][v] = (i, j)

        # Decode path traceback to aligned grapheme/phoneme chunks
        i, j = source_length, target_length
        source_chunks, target_chunks = [], []

        """
        print(np.round(alpha, 2))
        print()

        print(source, target)
        for row in traceback:
            print(row)
        print()
        print(i, j)
        """

        while i > 0 or j > 0:
            # print(i, j)

            # if i == 0 or j == 0:
            #    raise

            s_offset, t_offset = traceback[i][j]

            if s_offset == 0 and t_offset == 0:
                raise RuntimeError(
                    f"Cannot align {source} and {target} with {self.max_source_ngram=} and {self.max_target_ngram=}"
                )

            source_chunks.append(source[i - s_offset:i])
            target_chunks.append(target[j - t_offset:j])

            i -= s_offset
            j -= t_offset

        # Split source, target chunks that have same length
        source_chunks = source_chunks[::-1]
        target_chunks = target_chunks[::-1]

        split_source_chunks, split_target_chunks = [], []
        for source_chunk, target_chunk in zip(source_chunks, target_chunks):
            if len(source_chunk) == len(target_chunk) and len(source_chunk) > 1:
                for s_char, t_char in zip(source_chunk, target_chunk):
                    split_source_chunks.append([s_char])
                    split_target_chunks.append([t_char])
            else:
                split_source_chunks.append(list(source_chunk))
                split_target_chunks.append(list(target_chunk))

        return split_source_chunks, split_target_chunks

    def align(self, source, target):
        alignments = []
        for s, t in zip(source, target):
            try:
                source_alignment, target_alignment = self._align_instance(s, t)
            except RuntimeError:
                continue

            alignments.append((source_alignment, target_alignment))

        return alignments
