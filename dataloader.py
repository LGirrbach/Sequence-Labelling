import torch

from torch import Tensor
from typing import List, Union, Tuple
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from vocabulary import SequenceLabellingVocabulary


def _collate_condition(condition_batch: Tuple[Tuple[Tensor, ...]]):
    if not condition_batch:
        return tuple()

    combined_condition = []
    for conditions in zip(*condition_batch):
        if len(conditions[0].shape) == 1:
            combined_condition.append(torch.stack(conditions))
        elif len(conditions[0].shape) == 2:
            combined_condition.append(pad_sequence(list(conditions), batch_first=True, padding_value=0))
        else:
            raise ValueError("Condition format not supported")

    return combined_condition


class SequenceLabellingDataset(Dataset):
    def __init__(self, vocabulary: SequenceLabellingVocabulary, x: List[Union[str, List[str]]],
                 y: List[Union[str, List[str]]] = None):
        assert y is None or len(x) == len(y)
        self._x = x
        self._y = y
        self.vocabulary = vocabulary

    def _get_condition(self, idx: int):
        return tuple()

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx: int):
        source = self.vocabulary.index_source(self._x[idx])
        condition = self._get_condition(idx)

        if self._y is not None:
            target = self.vocabulary.index_target(self._y[idx])
            return source, target, condition

        else:
            return source, condition

    @staticmethod
    def train_batch_collate(batch):
        source, target, condition = zip(*batch)
        lengths = torch.LongTensor([len(word) for word in source]).cpu()

        source = pad_sequence(source, batch_first=True, padding_value=0)
        if all(isinstance(t, Tensor) for t in target):
            target = pad_sequence(target, batch_first=True, padding_value=0)

        condition = _collate_condition(condition_batch=condition)

        return source, target, lengths, condition

    @staticmethod
    def eval_batch_collate(batch):
        source, condition = zip(*batch)
        lengths = torch.LongTensor([len(word) for word in source]).cpu()
        source = pad_sequence(source, batch_first=True, padding_value=0)
        condition = _collate_condition(condition_batch=condition)

        return source, lengths, condition
