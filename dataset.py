import torch

from typing import Iterable
from typing import Optional
from collections import namedtuple
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from vocabulary import SequenceLabellingVocabulary

Batch = namedtuple(
    "Batch", field_names=["sources", "targets", "source_lengths", "target_lengths", "raw_sources", "raw_targets"]
)

RawDataset = namedtuple("RawDataset", ["sources", "targets"])
RawBatchElement = namedtuple("RawBatchElement", ["source", "target"])


class SequenceLabellingDataset(Dataset):
    def __init__(self, dataset: RawDataset, source_vocabulary: SequenceLabellingVocabulary,
                 target_vocabulary: Optional[SequenceLabellingVocabulary] = None):
        super(SequenceLabellingDataset, self).__init__()

        self.sources = dataset.sources
        self.targets = dataset.targets
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary

        if self.targets is not None:
            assert len(self.sources) == len(self.targets)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> RawBatchElement:
        if self.targets is not None:
            return RawBatchElement(source=self.sources[idx], target=self.targets[idx])
        else:
            return RawBatchElement(source=self.sources[idx], target=None)

    def collate_fn(self, batch: Iterable[RawBatchElement]) -> Batch:
        # Collect sources and targets
        sources = [batch_element.source for batch_element in batch]
        targets = [batch_element.target for batch_element in batch]

        # Index sources and targets
        indexed_sources = [self.source_vocabulary.index_sequence(source) for source in sources]
        indexed_sources = [torch.tensor(source).long() for source in indexed_sources]
        indexed_sources = pad_sequence(indexed_sources, batch_first=True, padding_value=0)
        source_lengths = [len(source) for source in sources]
        source_lengths = torch.tensor(source_lengths).long()

        if all([target is not None for target in targets]):
            indexed_targets = [self.target_vocabulary.index_sequence(target) for target in targets]
            indexed_targets = [torch.tensor(target).long() for target in indexed_targets]
            indexed_targets = pad_sequence(indexed_targets, batch_first=True, padding_value=0)
            target_lengths = [len(target) for target in targets]
            target_lengths = torch.tensor(target_lengths).long()
        else:
            indexed_targets = None
            target_lengths = None
            targets = None

        return Batch(
            sources=indexed_sources, targets=indexed_targets, source_lengths=source_lengths,
            target_lengths=target_lengths, raw_sources=sources, raw_targets=targets
        )
