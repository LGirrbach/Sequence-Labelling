# First take care of torch related inputs
import torch
import torch.nn as nn

from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import OneCycleLR

# Then import the rest
import os
import numpy as np

from typing import List
from tqdm.auto import tqdm
from torch_utils import torch_index
from torch_utils import softmax_2d
from torch_utils import make_mask_3d
from logger import logger as logging
from util import exponential_moving_avg as update_loss
from preprocessing.aligners.neural_aligner.kernels import Kernel
from preprocessing.aligners.neural_aligner.encoders import Encoder
from preprocessing.aligners.neural_aligner.viterbi_align import viterbi_align


def collate_fn(batch):
    source, target = zip(*batch)
    return source, target


class SimilarityMatrix(nn.Module):
    """
    Given 2 sequences (here: graphemes and phonemes), calculate (unnormalised) alignment/similarity
    scores for each pair of elements of the 2 sequences, e.g.
            b1    b2  ...
      a1   v11   v12  ...
      a2   v21   v22  ...
      ...  ...   ...  ...
    """

    def __init__(self, embedding_dim: int, source_vocab_size: int, target_vocab_size: int,
                 kernel: str = 'linear', kernel_parameters: dict = None, encoder: str = 'conv',
                 source_encoder_parameters: dict = None, target_encoder_parameters: dict = None):
        super().__init__()

        # Initialise embedding layers
        self.embedding_dim = embedding_dim
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_embed = nn.Embedding(source_vocab_size, self.embedding_dim, padding_idx=0, max_norm=10.0)
        self.target_embed = nn.Embedding(target_vocab_size, self.embedding_dim, padding_idx=0, max_norm=10.0)

        # Initialise source and target encoders
        self.source_encoder = Encoder.encoder_factory(encoder, embedding_dim, **source_encoder_parameters)
        self.target_encoder = Encoder.encoder_factory(encoder, embedding_dim, **target_encoder_parameters)

        # Assert source and target representations have same dimensionality
        if self.source_encoder.output_size != self.target_encoder.output_size:
            s_size = self.source_encoder.output_size
            t_size = self.target_encoder.output_size
            raise RuntimeError(f"Source encoder hidden size ({s_size}) != Target encoder hidden size ({t_size})")

        # Initialise similarity kernel (for computing raw alignment scores)
        kernel_parameters["num_features"] = self.source_encoder.output_size
        self.kernel = Kernel.kernel_factory(kernel, **kernel_parameters)

    def encoder(self, source: torch.Tensor, target: torch.Tensor, source_lengths: torch.Tensor = None,
                target_lengths: torch.Tensor = None):
        """Encodes source and target sequences"""
        # Embed graphemes / phonemes
        source_embedding = self.source_embed(source)
        target_embedding = self.target_embed(target)

        # Encode graphemes / phonemes
        source_encoded = self.source_encoder(source_embedding, lengths=source_lengths)
        target_encoded = self.target_encoder(target_embedding, lengths=target_lengths)

        return source_encoded, target_encoded

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        # If processing minibatch, get lengths
        assert len(source.shape) == len(target.shape)

        if len(source.shape) == 2:
            source_lengths = (source != 0).sum(dim=1).long().flatten()
            target_lengths = (target != 0).sum(dim=1).long().flatten()

        else:
            source_lengths, target_lengths = None, None

        # Calculate encoded source and target
        source_encoded, target_encoded = self.encoder(source, target, source_lengths, target_lengths)
        # Calculate similarity scores by kernel
        scores = self.kernel(source_encoded, target_encoded)

        # Mask padding scores if processing minibatch
        if len(source.shape) == 2:
            mask = make_mask_3d(source_lengths, target_lengths)
            scores = torch.masked_fill(scores, mask=mask, value=-np.inf)
        else:
            mask = None

        result_dict = {
            'source': source,
            'source_lengths': source_lengths,
            'source_encoded': source_encoded,
            'target': target,
            'target_lengths': target_lengths,
            'target_encoded': target_encoded,
            'scores': scores,
            'mask': mask
        }

        return result_dict


class NeuralAligner:
    """
    Base class for neural many-to-many alignments.
    Implements:
      * Building grapheme / phoneme vocabularies and indexing
      * Grapheme / phoneme encoding
      * Building the grapheme - phoneme similarity matrix (raw = unnormalised scores)
    """

    def __init__(self, embedding_dim: int, batch_size: int = 64, epochs: int = 5, lr: float = 0.1,
                 kernel: str = 'linear', kernel_parameters: dict = None, encoder: str = 'conv',
                 source_encoder_parameters: dict = None, target_encoder_parameters: dict = None, **params):

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.is_initialised = False

        self.embedding_dim = embedding_dim
        self.kernel = kernel
        self.kernel_parameters = kernel_parameters
        self.encoder = encoder
        self.source_encoder_parameters = source_encoder_parameters
        self.target_encoder_parameters = target_encoder_parameters
        self.params = params

        # Placeholder variable for model
        self.model: nn.Module = nn.Module()

        # Placeholder variables for vocab / indexers
        self.source_vocab = None
        self.target_vocab = None
        self.source2idx = None
        self.target2idx = None
        self.idx2source = None
        self.idx2target = None

    def _init_train_data(self, source: List[str], target: List[List[str]]):
        """Generates vocabs and indexes data"""

        # Generate grapheme / phoneme vocabularies
        self.source_vocab = list(sorted(set.union(*(set(s) for s in source))))
        self.target_vocab = list(sorted(set.union(*(set(t) for t in target))))

        # Generate grapheme / phoneme indexers
        self.source2idx = {char: idx + 1 for idx, char in enumerate(self.source_vocab)}
        self.target2idx = {char: idx + 1 for idx, char in enumerate(self.target_vocab)}
        self.idx2source = {idx: char for char, idx in self.source2idx.items()}
        self.idx2target = {idx: char for char, idx in self.target2idx.items()}

        # Index graphemes / phonemes
        indexed_source = [torch_index(s, self.source2idx) for s in source]
        indexed_target = [torch_index(t, self.target2idx) for t in target]
        assert len(indexed_source) == len(indexed_target)

        train_data = list(zip(indexed_source, indexed_target))
        return train_data

    def save(self, path):
        logging.info(f"Saving model to {path}")
        save_info = {
            'model_state_dict': self.model.state_dict(),
            'source_vocab': self.source_vocab,
            'target_vocab': self.target_vocab,
            'source2idx': self.source2idx,
            'target2idx': self.target2idx,
            'idx2source': self.idx2source,
            'idx2target': self.idx2target
        }

        torch.save(save_info, os.path.join(path, "aligner.pt"))
        logging.info("Successfully saved model.")

    def load(self, path):
        model_info = torch.load(path)
        self.source_vocab = model_info['source_vocab']
        self.target_vocab = model_info['target_vocab']
        self.source2idx = model_info['source2idx']
        self.target2idx = model_info['target2idx']
        self.idx2source = model_info['idx2source']
        self.idx2target = model_info['idx2target']

        self._initialise_model()
        self.model.load_state_dict(model_info['model_state_dict'])
        logging.info(f"Successfully loaded model from {path}")

    def _initialise_model(self):
        """Placeholder method for model initialisation"""
        raise NotImplementedError

    def _get_loss(self, batch):
        """Placeholder method for loss calculation"""
        raise NotImplementedError

    def fit(self, source: List[str], target: List[List[str]]):
        """Train model on given grapheme / phoneme pairs"""
        # Can only train uninitialised model
        assert not self.is_initialised
        # Index train data and build vocabularies
        logging.info("Indexing training data")
        train_data = self._init_train_data(source, target)

        # Initialise model
        logging.info("Initialise model")
        self._initialise_model()

        # Define data loader, optimizer, and learning rate scheduler
        dataset = DataLoader(train_data, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)
        optimizer = SGD(self.model.parameters(), lr=self.lr, weight_decay=0.00)
        scheduler = OneCycleLR(optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=len(dataset))

        self.is_initialised = True

        # Train model
        logging.info("Start training")
        running_loss = None
        pbar = tqdm(total=self.epochs * len(dataset), desc="Training progress:")
        self.model.train()
        for epoch in range(self.epochs):
            for batch in dataset:
                # Reset gradients
                optimizer.zero_grad()
                # Calculate loss
                loss = self._get_loss(batch)

                # Perform optimisation step
                loss.backward()
                clip_grad_value_(self.model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()

                # Display loss
                detached_loss = loss.detach().cpu().item()
                running_loss = update_loss(running_loss, detached_loss)

                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {detached_loss:.2f}, Running Loss: {running_loss:.4f}")

        pbar.close()
        logging.info("Finished training")

    def _align_instance(self, source: torch.Tensor, target: torch.Tensor):
        # Calculate raw scores from model kernel
        raw_scores = self.model(source, target)['scores']

        # Normalise scores
        # Currently, a normalisation of the complete joint seems to work better than row or column-wise
        # normalisation, but this is up to future exploration
        scores = softmax_2d(raw_scores, log=True)

        # Convert everything to numpy
        raw_scores = raw_scores.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        source = source.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        # Generate alignment
        source_alignment, target_alignment = viterbi_align(source, target, scores)
        return source_alignment, target_alignment, (raw_scores, scores)

    def align(self, source, target):
        # Index graphemes / phonemes
        indexed_source = [torch_index(s, self.source2idx) for s in source]
        indexed_target = [torch_index(t, self.target2idx) for t in target]

        # Define data loader
        data = list(zip(indexed_source, indexed_target))

        # Generate alignments for all grapheme / phoneme pairs
        self.model.eval()
        with torch.no_grad():
            alignments = [self._align_instance(s, t) for s, t in tqdm(data, desc="Alignment progress:")]

        # Convert alignments to string (human-readable)
        restored_alignments = []
        for s, t, info in alignments:
            s = [[self.idx2source.get(idx, '-') for idx in c] for c in s]
            t = [[self.idx2target.get(idx, '-') for idx in c] for c in t]

            restored_alignments.append((s, t, info))

        return restored_alignments


class NeuralEMAligner(NeuralAligner):
    def _initialise_model(self):
        """Placeholder method for model initialisation"""
        self.model = SimilarityMatrix(embedding_dim=self.embedding_dim, source_vocab_size=len(self.source_vocab) + 1,
                                      target_vocab_size=len(self.target_vocab) + 1, kernel=self.kernel,
                                      kernel_parameters=self.kernel_parameters, encoder=self.encoder,
                                      source_encoder_parameters=self.source_encoder_parameters,
                                      target_encoder_parameters=self.target_encoder_parameters)

    def _get_loss(self, batch):
        """Placeholder method for loss calculation"""
        source_batch, target_batch = batch
        source_batch = pad_sequence(source_batch, batch_first=True, padding_value=0)
        target_batch = pad_sequence(target_batch, batch_first=True, padding_value=0)

        # Calculate alignment scores
        batch_info = self.model(source_batch, target_batch)
        scores = batch_info['scores']

        # Extract separate alignment score matrices for each batch element
        source_lengths = batch_info['source_lengths'].tolist()
        target_lengths = batch_info['target_lengths'].tolist()
        indices = list(zip(source_lengths, target_lengths))
        scores = [scores[i, :s_length, :t_length] for i, (s_length, t_length) in enumerate(indices)]

        # Calculate -log p(source, target) for each batch element
        losses = [self._get_instance_loss(alignment_scores.contiguous()) for alignment_scores in scores]

        # Return mean negative log likelihood of batch
        loss = torch.stack(losses).mean()
        return loss

    @staticmethod
    def _get_instance_loss(scores: torch.Tensor):
        """
        Calculate probability p(source, target) as the sum probabilities of all alignment paths
        using dynamic programming.

        Probabilities are calculated from the pairwise alignments scores of graphemes and phonemes.
        Calculations are performed using log-probabilities for numerical stability.

        Optimisation target is to minimise the negative log-likelihood log p(source, target)
        """
        source_length, target_length = scores.shape
        # Normalise scores
        # Currently, a normalisation of the complete joint seems to work better than row or column-wise
        # normalisation, but this is up to future exploration
        scores = softmax_2d(scores, log=True)

        # Initialise alignment scores:
        # `alpha` stores the marginal probabilities of aligning each grapheme-phoneme pair, that is
        # the probability of any alignment path going through the respective alignment
        alpha = [[torch.tensor(0.0) for _ in range(target_length)] for _ in range(source_length)]
        # Since we don't allow deletion, the first grapheme and phoneme have to be aligned,
        # therefore we can initialise the upper left corner with their alignment probability
        alpha[0][0] = scores[0, 0]

        # Initialise first column (alignments of all first phoneme)
        for i in range(1, source_length):
            alpha[i][0] = alpha[i - 1][0] + scores[i, 0]

        # Initialise first row (alignments of first grapheme)
        for j in range(1, target_length):
            alpha[0][j] = alpha[0][j - 1] + scores[0, j]

        # Calculate scores for all other grapheme-phoneme pairs
        for i in range(1, source_length):
            for j in range(1, target_length):
                # We allow the following operations:
                #  1. Align current grapheme to previous phoneme
                #  2. Align current phoneme to previous grapheme
                #  3. Align current grapheme to current phoneme
                prev_scores = [
                    alpha[i - 1][j - 1], alpha[i - 1][j], alpha[i][j - 1]
                ]
                prev_scores = torch.stack(prev_scores)
                prev_scores = prev_scores + scores[i, j]
                alpha[i][j] = torch.logsumexp(prev_scores, dim=0)

        # We need to maximise the probability p(s, t) of any alignment
        # The NLL is stored in the lower right corner (probability of any path moving through aligning the
        # last grapheme with the last phoneme)
        loss = -alpha[source_length - 1][target_length - 1]
        return loss
