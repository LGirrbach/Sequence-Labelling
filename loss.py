import torch

from torch import Tensor
from dataset import Batch
from model import LSTMModel
from util import make_mask_2d
from collections import namedtuple
from torch.nn.functional import cross_entropy
from torch.nn.functional import ctc_loss as ctc

ModelOutput = namedtuple("ModelOutput", field_names=["loss", "logits"])


def _get_logits(model: LSTMModel, batch: Batch) -> Tensor:
    return model(batch.sources.to(model.device), batch.source_lengths)


def cross_entropy_loss(model: LSTMModel, batch: Batch, reduction: str = "mean") -> ModelOutput:
    logits = _get_logits(model=model, batch=batch)
    logits = torch.flatten(logits, end_dim=-2)
    targets = torch.flatten(batch.targets).to(logits.device)
    loss = cross_entropy(logits, targets, ignore_index=0, reduction=reduction)
    return ModelOutput(loss=loss, logits=logits)


def ctc_loss(model: LSTMModel, batch: Batch, reduction: str = "mean") -> ModelOutput:
    logits = _get_logits(model=model, batch=batch)
    log_probs = torch.log_softmax(logits, dim=-1)
    log_probs = torch.transpose(log_probs, 0, 1)
    targets = batch.targets.to(logits.device)
    tau = model.tau
    loss = ctc(
        log_probs=log_probs, targets=targets, input_lengths=tau * batch.source_lengths,
        target_lengths=batch.target_lengths, blank=0, reduction=reduction
    )
    return ModelOutput(loss=loss, logits=logits)


def crf_loss(model: LSTMModel, batch: Batch, reduction: str = "mean") -> ModelOutput:
    logits = _get_logits(model=model, batch=batch)
    targets = batch.targets.to(logits.device)
    # emission scores shape: batch x timesteps x #labels
    # tags: batch x timesteps

    batch_size = logits.shape[0]

    # Apply log-softmax
    emission_scores = torch.log_softmax(logits, dim=-1)

    # Extract emission scores
    tag_index = targets.unsqueeze(2)
    emission_scores = torch.gather(emission_scores, index=tag_index, dim=2)
    emission_scores = emission_scores.squeeze(2)
    # Shape [Batch, Timesteps]

    # Extract transition scores
    transition_scores = model.crf.transition_scores[batch.targets[:, 1:], batch.targets[:, :-1]]

    # Extract prior
    prior = model.crf.prior[batch.targets[:, 0]].contiguous()
    prior = prior.reshape((batch_size, 1))

    # Combine prior and transition scores
    transition_scores = torch.cat([prior, transition_scores], dim=1)

    # Calculate transition probabilities to stop tag
    length_index = (batch.source_lengths - 1).unsqueeze(1).to(emission_scores.device)
    final_tags = torch.gather(targets, index=length_index, dim=1)
    final_tags = final_tags.flatten()
    final_transition_scores = model.crf.final_transition_scores[final_tags]
    final_transition_scores = final_transition_scores.contiguous()

    # Calculate path probabilities
    nll = transition_scores + emission_scores
    # Mask padding
    mask = make_mask_2d(batch.source_lengths).to(emission_scores.device)
    nll = torch.masked_fill(nll, mask=mask, value=0.0)

    # Sum tag scores for each sequence
    # Add transition probabilities to stop tag
    nll = -(torch.sum(nll, dim=1) + final_transition_scores)

    if reduction == "mean":
        nll = torch.mean(nll, dim=0)
    elif reduction == "sum":
        nll = torch.sum(nll, dim=0)
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    return ModelOutput(loss=nll, logits=logits)
