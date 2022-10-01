import torch
import numpy as np

from typing import List
from torch import Tensor
from model import LSTMModel
from collections import namedtuple
from vocabulary import SequenceLabellingVocabulary

AlignmentPosition = namedtuple("AlignmentPosition", ["symbol", "predictions"])
Prediction = namedtuple("TransducerPrediction", ["prediction", "alignment"])


def _convert_idx(sources: List[List[str]], predictions: List[List[int]],
                 target_vocabulary: SequenceLabellingVocabulary, tau: int) -> List[Prediction]:
    aligned_predictions = []

    for source, prediction in zip(sources, predictions):
        prediction = np.array(prediction).reshape((-1, tau)).tolist()
        assert len(prediction) == len(source)

        alignment = []
        decoded_prediction = []
        for source_symbol, aligned_idx in zip(source, prediction):
            decoded_symbols = target_vocabulary.convert_idx(aligned_idx)
            decoded_symbols = [symbol for symbol in decoded_symbols if symbol != target_vocabulary.PAD_TOKEN]
            alignment.append(AlignmentPosition(symbol=source_symbol, predictions=decoded_symbols))
            decoded_prediction.extend(decoded_symbols)

        aligned_predictions.append(Prediction(prediction=decoded_prediction, alignment=alignment))

    return aligned_predictions


def argmax_decode(model: LSTMModel, logits: Tensor, lengths: Tensor, sources: List[List[str]],
                  target_vocabulary: SequenceLabellingVocabulary, tau: int) -> List[Prediction]:
    predictions = torch.argmax(logits, dim=-1).detach().cpu().tolist()
    lengths = (tau * lengths).detach().cpu().tolist()
    predictions = [prediction[:length] for prediction, length in zip(predictions, lengths)]

    return _convert_idx(sources=sources, predictions=predictions, target_vocabulary=target_vocabulary, tau=tau)


def viterbi_decode(model: LSTMModel, logits: Tensor, lengths: Tensor, sources: List[List[str]],
                   target_vocabulary: SequenceLabellingVocabulary, tau: int) -> List[Prediction]:
    lengths = tau * lengths
    # Get relevant dimension info
    batch, timesteps, num_tags = logits.shape

    # Apply log-softmax
    emission_scores = torch.log_softmax(logits, dim=-1)

    # Normalise transition and emission scores
    transition_scores = model.crf.transition_scores.T

    # Calculate prior
    prior = model.crf.prior.unsqueeze(0).expand(batch, num_tags)
    final_transition_scores = model.crf.final_transition_scores.unsqueeze(0).expand(batch, num_tags)

    # Start with emission scores at first time step times prior
    prev_alpha = emission_scores[:, 0, :].contiguous()
    prev_alpha = prior + prev_alpha
    alpha = [prev_alpha]

    # Initialise back-pointers
    back_pointers = torch.zeros(batch, timesteps, num_tags).long()

    # Forward recursion
    for t in range(1, timesteps):
        emission_scores_t = emission_scores[:, t, :]
        # Shape [Batch, 1, #Tags]
        prev_alpha = prev_alpha.unsqueeze(1)
        # Shape [Batch, #Tags, 1]

        alpha_t = prev_alpha + transition_scores
        # Shape [Batch, #Tags, #Tags]
        # Get maximum values for each tag
        alpha_t, back_pointers_t = torch.max(alpha_t, dim=2)
        alpha_t = alpha_t + emission_scores_t
        alpha.append(alpha_t)
        back_pointers[:, t, :] = back_pointers_t
        prev_alpha = alpha_t

    alpha = torch.stack(alpha)
    alpha = alpha.transpose(0, 1)

    viterbi_scores = alpha[torch.arange(batch), lengths - 1]
    viterbi_scores = viterbi_scores + final_transition_scores

    viterbi_scores, start_tag_indices = torch.max(viterbi_scores, dim=1)

    # Reconstruct predicted paths from back-pointers
    start_tag_indices = start_tag_indices.tolist()
    back_pointers = back_pointers.detach().cpu().numpy()
    predicted_paths = []

    for batch_idx, length in enumerate(lengths.tolist()):
        if length == 0:
            predicted_paths.append([])
            continue

        start_idx = start_tag_indices[batch_idx]
        predicted_path = [start_idx]

        for t in range(1, length):
            start_idx = back_pointers[batch_idx, length - t, start_idx].item()
            predicted_path.append(start_idx)

        predicted_path = list(reversed(predicted_path))
        try:
            assert len(predicted_path) == length
        except AssertionError:
            print(predicted_path)
            print(length)
            raise

        predicted_paths.append(predicted_path)

    # Decode predictions
    return _convert_idx(sources=sources, predictions=predicted_paths, target_vocabulary=target_vocabulary, tau=tau)


def ctc_crf_decode(model: LSTMModel, logits: Tensor, lengths: Tensor, sources: List[List[str]],
                   target_vocabulary: SequenceLabellingVocabulary, tau: int) -> List[Prediction]:
    lengths = tau * lengths
    # Get relevant dimension info
    batch, timesteps, num_tags = logits.shape

    # Get relevant scores
    batch_emission_scores = torch.log_softmax(logits, dim=-1)
    transition_scores = model.crf.transition_scores.T
    prior = model.crf.prior
    final_transition_scores = model.crf.final_transition_scores

    predicted_paths = []

    for length, emission_scores in zip(lengths, batch_emission_scores):
        length = length.detach().cpu().item()
        emission_scores = emission_scores[:length]

        alpha = torch.empty(0, num_tags, device=batch_emission_scores.device)
        backpointers_time = []
        backpointers_label = []

        for t, emission_scores_t in enumerate(emission_scores):
            best_prev_label = torch.full((num_tags,), fill_value=-1, dtype=torch.long, device=alpha.device)
            best_prev_timestep = torch.full((num_tags,), fill_value=-1, dtype=torch.long, device=alpha.device)
            best_score = emission_scores_t + prior + emission_scores[:t, 0].sum()

            if t > 0:
                blank_scores = emission_scores[:t, 0]
                blank_scores = blank_scores.sum() - blank_scores.cumsum(dim=0)

                scores = (
                    alpha.unsqueeze(2).expand((t, num_tags, num_tags)) +
                    blank_scores.reshape(-1, 1, 1).expand((t, num_tags, num_tags)) +
                    emission_scores_t.reshape(1, 1, -1).expand((t, num_tags, num_tags)) +
                    transition_scores.unsqueeze(0).expand((t, num_tags, num_tags))
                )

                scores, s = torch.max(scores, dim=0)
                scores, prev_label = torch.max(scores, dim=0)
                s = s[prev_label, torch.arange(num_tags)]

                superior_idx = torch.nonzero(scores > best_score, as_tuple=True)
                best_score[superior_idx] = scores[superior_idx]
                best_prev_label[superior_idx] = prev_label[superior_idx]
                best_prev_timestep[superior_idx] = s[superior_idx]

            alpha = torch.cat([alpha, best_score.reshape(1, -1)], dim=0)
            backpointers_time.append(best_prev_timestep.detach().cpu().tolist())
            backpointers_label.append(best_prev_label.detach().cpu().tolist())

        blank_scores = emission_scores[:, 0].sum() - emission_scores[:, 0].cumsum(dim=0)
        blank_scores = blank_scores.unsqueeze(1).expand((length, num_tags))
        final_scores = alpha + final_transition_scores.unsqueeze(0).expand((length, num_tags)) + blank_scores
        final_scores, best_end_timestep = torch.max(final_scores, dim=0)
        best_score, best_end_label = torch.max(final_scores, dim=0)

        label = best_end_label.cpu().item()
        timestep = best_end_timestep[best_end_label].cpu().item()

        predicted_path = [0 for _ in range(length)]

        while timestep != -1:
            predicted_path[timestep] = label
            timestep, label = backpointers_time[timestep][label], backpointers_label[timestep][label]

        predicted_paths.append(predicted_path)

    # Decode predictions
    return _convert_idx(sources=sources, predictions=predicted_paths, target_vocabulary=target_vocabulary, tau=tau)
