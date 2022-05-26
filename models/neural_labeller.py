import os
import time
import torch
import numpy as np
import torch.nn as nn

from abc import ABC
from tqdm import tqdm
from typing import Any
from typing import Dict
from typing import List
from torch import Tensor
from typing import Union
from typing import Tuple
from logger import logger
from torch.optim import AdamW
from metrics import edit_distance
from metrics import token_accuracy
from torch_utils import move_to_cpu
from torch.optim import lr_scheduler
from torch_utils import move_to_cuda
from metrics import sequence_accuracy
from torch.utils.data import DataLoader
from util import exponential_moving_avg
from torch.nn.utils import clip_grad_value_
from constants import Sequence, SequenceData
from metrics import normalised_edit_distance
from dataloader import SequenceLabellingDataset
from dataloader import SequenceLabellingVocabulary


class NeuralLabeller(ABC):
    def __init__(self, model: nn.Module = None, use_cuda: bool = False, batch_size: int = 32, epochs: int = 1,
                 lr: float = 0.001,  weight_decay: float = 0.01, gradient_clip_value: float = None,
                 early_stopping: bool = False, tolerance: int = 3, min_epochs: int = 1,
                 main_metric: str = "edit-distance", maximise_metric: bool = False, num_dataloader_workers: int = 6,
                 save_path: str = None, name: str = None):
        super(NeuralLabeller, self).__init__()

        # Save arguments
        self.model = model
        self.cuda = use_cuda and torch.cuda.is_available()
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip_value = gradient_clip_value
        self.early_stopping = early_stopping
        self.tolerance = tolerance
        self.min_epochs = min_epochs
        self.main_metric = main_metric
        self.maximise_metric = maximise_metric
        self.num_dataloader_workers = num_dataloader_workers
        self.save_path = save_path
        self.name = name

        # Derived variables
        self.tolerance_countdown = self.tolerance

        # Placeholder variables
        self.optimizer = None
        self.scheduler = None
        self.vocabulary = None
        self.best_dev_score = None
        self.metrics = None
        self.running_loss = None
        self.progress_bar = None
        self.update_lr_every_step = False

    def get_params(self) -> Dict:
        return {
            'use_cuda': self.cuda,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'gradient_clip_value': self.gradient_clip_value,
            'early_stopping': self.early_stopping,
            'tolerance': self.tolerance,
            'min_epochs': self.min_epochs,
            'main_metric': self.main_metric,
            'maximise_metric': self.main_metric,
            'num_dataloader_workers': self.num_dataloader_workers,
            'save_path': self.save_path,
            'name': self.name
        }

    def save(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        save_dict = {
            'encoder_type': type(self.model['encoder']),
            'encoder_params': self.model['encoder'].get_save_info(),
            'model_state_dict': self.model.state_dict(),
            'training_params': self.get_params(),
            'vocabulary': self.vocabulary,
            'model_type': type(self),
            'best_dev_score': self.best_dev_score,
            'metrics': self.metrics,
            'running_loss': self.running_loss,
            'tolerance_countdown': self.tolerance_countdown
        }

        name = self.name if self.name is not None else 'labeller'
        torch.save(save_dict, os.path.join(self.save_path, name + ".pt"))

    @staticmethod
    def load(path: str):
        """Load model from file"""
        # Load data from file
        param_dict = torch.load(path, map_location='cpu')
        # Instantiate encoder (pytorch model)
        encoder = param_dict['encoder_type'].load(param_dict['encoder_params'])
        # Instantiate model
        model = param_dict['model_type'](encoder, **param_dict['training_params'])
        # Load vocabulary
        model.vocabulary = param_dict['vocabulary']
        # Initialise encoder model
        model.build_model()
        # Move encoder to cpu (necessary s.t. we don't clash with the loaded state dict, which is mapped to cpu)
        model.model = model.model.cpu()
        # Load encoder parameters
        model.model.load_state_dict(param_dict['model_state_dict'])

        # Load training vars
        model.best_dev_score = param_dict['best_dev_score']
        model.metrics = param_dict['metrics']
        model.running_loss = param_dict['running_loss']
        model.tolerance_countdown = param_dict['tolerance_countdown']

        return model

    def build_model(self) -> nn.Module:
        """Initialise encoder model"""
        raise NotImplementedError

    def _calculate_loss(self, x_batch: Tensor, y_batch: Tensor, lengths: Tensor, condition: Any) -> Tensor:
        """Calculate loss for minibatch (implement in subclass)"""
        raise NotImplementedError("Overwrite _calculate_loss in subclass")

    def _decode_prediction(self, prediction_scores: Tensor, length: List[int]) -> Sequence:
        """
        Predict labels from label prediction scores calculated by encoder.
        Default implementation is greedy decoding.
        """
        batch_predictions = []

        for prediction_scores, length in zip(prediction_scores, length):
            length = int(length)
            # Truncate scores
            prediction_scores = prediction_scores[:length]
            # Take maximum likelihood predictions
            predicted_indices = torch.argmax(prediction_scores, dim=-1)
            predicted_indices = predicted_indices.detach().cpu().tolist()
            # Decode labels
            predicted_labels = self.vocabulary.decode_target(predicted_indices)
            batch_predictions.append(predicted_labels)

        return batch_predictions

    def postprocess_prediction(self, prediction: Sequence) -> Sequence:
        """Post-process predicted labels"""
        return [("" if label in self.vocabulary.SPECIALS else label) for label in prediction]

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _build_lr_scheduler(self, dataset: DataLoader):
        mode = "max" if self.maximise_metric else "min"
        self.update_lr_every_step = False
        return lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=mode, patience=self.tolerance, factor=0.5)

    def _step(self, x_batch: Tensor, y_batch: Tensor, lengths: Tensor, condition: Any) -> float:
        """Perform training step: Calculate loss, differentiate and update parameters"""
        self.optimizer.zero_grad()  # Clear gradients

        # Calculate loss (implemented in subclass)
        loss = self._calculate_loss(x_batch, y_batch, lengths, condition)

        # Run automatic differentiation
        loss.backward()
        # Clip gradient values (optional)
        if self.gradient_clip_value is not None and self.gradient_clip_value > 0:
            clip_grad_value_(self.model.parameters(), self.gradient_clip_value)

        # Optimisation step
        self.optimizer.step()

        # LR scheduler step
        if self.scheduler is not None and self.update_lr_every_step:
            self._update_lr_scheduler()

        return loss.detach().cpu().item()

    def _prepare_train_datasets(self, x_train: SequenceData = None, y_train: SequenceData = None,
                                vocabulary: SequenceLabellingVocabulary = None,
                                train_data: SequenceLabellingDataset = None):
        # Check train arguments:
        # The following options exist:
        #  1. Provide x_train and y_train -> Build vocabulary and data loader
        #  2. Provide x_train, y_train, and vocabulary -> Build data loader
        #  3. Provide data loader and vocabulary -> use provided data loader and vocabulary
        if x_train is not None and y_train is not None:
            assert len(x_train) == len(y_train)
            assert train_data is None, "Cannot provide both x_train, y_train and data loader"
            # Make vocabulary
            if vocabulary is None:
                vocabulary = SequenceLabellingVocabulary(x_train, y_train)

            # Make data loader
            train_data = SequenceLabellingDataset(vocabulary, x_train, y_train)

        elif train_data is not None and vocabulary is not None:
            assert x_train is None and y_train is None, "Cannot provide both x_train, y_train and data loader"

        else:
            error_msg = \
                "You must provide either x_train, y_train or x_train, y_train, vocabulary or train_data, vocabulary"
            raise ValueError(error_msg)

        # Make batched, iterable dataset
        train_dataloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            collate_fn=train_data.train_batch_collate,
            num_workers=self.num_dataloader_workers,
            shuffle=True
        )

        # Sanity checks
        assert train_data is not None
        assert vocabulary is not None
        assert train_dataloader is not None

        return train_data, vocabulary, train_dataloader

    def _prepare_dev_data(self, x_dev: SequenceData = None, y_dev: SequenceData = None,
                          dev_data: SequenceLabellingDataset = None):
        # Check dev arguments:
        # The following options are possible:
        #  1. Provide x_dev and y_dev -> Build data containing only x_dev
        #  2. Provide data loader and y_dev, -> Use provided data loader
        if x_dev is not None and y_dev is not None:
            assert dev_data is None, "Cannot provide both x_dev, y_dev and dev data loader"
            dev_data = SequenceLabellingDataset(self.vocabulary, x=x_dev)

        elif dev_data is not None:
            assert y_dev is not None
            assert x_dev is None, "Cannot provide both x_dev, y_dev and dev data loader"

        else:
            y_dev = None
            dev_data = None

        return dev_data, y_dev

    def _count_model_parameters(self) -> Tuple[int, int]:
        total_num_parameters = 0
        total_num_trainable_parameters = 0

        for parameter in self.model.parameters():
            total_num_parameters += parameter.numel()
            if parameter.requires_grad:
                total_num_trainable_parameters += parameter.numel()

        return total_num_parameters, total_num_trainable_parameters

    def _update_running_loss(self, loss: float) -> None:
        self.running_loss = exponential_moving_avg(self.running_loss, loss)

    def _update_progress_bar_info(self) -> None:
        msg = f"Running loss: {self.running_loss:.4f}"
        if self.best_dev_score is not None:
            msg += f" || {self.main_metric}: {self.best_dev_score:.3f}"

        msg += f" || LR: {np.mean(self.optimizer.param_groups[0]['lr']):.6f}"

        self.progress_bar.set_postfix_str(msg)

    def _update_lr_scheduler(self) -> None:
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            if self.best_dev_score is not None:
                self.scheduler.step(self.best_dev_score)
        else:
            self.scheduler.step()

    def _train_epoch(self, train_dataloader: DataLoader) -> None:
        self.model.train()
        # Iterate through train set
        for x_batch, y_batch, lengths, condition in train_dataloader:
            # Perform update for batch
            loss = self._step(x_batch, y_batch, lengths, condition)
            # Update loss statistics
            self._update_running_loss(loss)
            # Update progress bar display
            self._update_progress_bar_info()
            # Display progress
            self.progress_bar.update(1)

    def fit(self, x_train: SequenceData = None, y_train: SequenceData = None,
            vocabulary: SequenceLabellingVocabulary = None, train_data: SequenceLabellingDataset = None,
            x_dev: SequenceData = None, y_dev: SequenceData = None, dev_data: SequenceLabellingDataset = None):
        # Prepare train data
        train_data, vocabulary, train_dataloader = self._prepare_train_datasets(
            x_train=x_train, y_train=y_train, vocabulary=vocabulary, train_data=train_data
        )
        self.vocabulary = vocabulary

        # Prepare dev data
        dev_data, y_dev = self._prepare_dev_data(x_dev=x_dev, y_dev=y_dev, dev_data=dev_data)

        # Log data info
        logger.info(f"Training set contains {len(train_data)} data-points")
        if dev_data is None:
            logger.info("No development set provided. Early stopping disabled")
        else:
            logger.info(f"Development set contains {len(dev_data)} data-points")

        # Prepare model
        self.model = self.build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_lr_scheduler(train_dataloader)

        num_model_parameters, num_trainable_parameters = self._count_model_parameters()
        logger.info(f"Model has {num_model_parameters} parameters (total)")
        logger.info(f"Model has {num_trainable_parameters} trainable parameters")

        # Train
        logger.info("Start training")
        time.sleep(0.01)

        self.progress_bar = tqdm(total=self.epochs * len(train_dataloader), desc="Training progress")

        for epoch in range(self.epochs):
            # Train for one epoch
            self._train_epoch(train_dataloader)

            # If development set provided, evaluate after every epoch
            if dev_data is not None:
                early_stopping = self._evaluate_on_dev_set(dev_data, y_dev)

                if self.early_stopping and early_stopping and epoch >= self.min_epochs:
                    logger.info(f"Early stopping after {epoch + 1} epochs (tolerance = {self.tolerance})")
                    time.sleep(0.01)
                    break

            # If development data is not provided, safe model after every epoch (optional)
            elif self.save_path is not None:
                self.save()

            # Update learning rate after epoch (optional)
            if not self.update_lr_every_step:
                self._update_lr_scheduler()

        self.progress_bar.close()

    @staticmethod
    def _calculate_metrics(predictions: SequenceData, targets: SequenceData):
        return {
            'sequence-accuracy': sequence_accuracy(predictions, targets),
            'token-accuracy': token_accuracy(predictions, targets),
            'edit-distance': edit_distance(predictions, targets),
            'normalised-edit-distance': normalised_edit_distance(predictions, targets)
        }

    def _metric_improved(self, new_metric: float, old_metric: float):
        if old_metric is None:
            return True

        if self.maximise_metric:
            return new_metric > old_metric
        else:
            return new_metric < old_metric

    def _evaluate_on_dev_set(self, dev_data: SequenceLabellingDataset, y_dev: SequenceData):
        dev_predictions = self.predict(eval_data=dev_data, verbose=False)
        dev_predictions = [self.postprocess_prediction(prediction) for prediction in dev_predictions]

        self.metrics = self._calculate_metrics(dev_predictions, y_dev)
        best_dev_score = self.metrics[self.main_metric]

        if self._metric_improved(best_dev_score, self.best_dev_score):
            self.best_dev_score = best_dev_score
            self.tolerance_countdown = self.tolerance

            if self.save_path is not None:
                self.save()

        else:
            self.tolerance_countdown -= 1

        if self.tolerance_countdown < 0:
            return True
        else:
            return False

    def predict(self, x: SequenceData = None, eval_data: SequenceLabellingDataset = None, verbose: bool = True):
        # Check arguments
        if x is not None:
            assert eval_data is None

        if x is not None:
            eval_data = SequenceLabellingDataset(vocabulary=self.vocabulary, x=x)
        elif eval_data is None:
            raise ValueError("You have to provide either input sequences or dataset instance")

        eval_dataloader = DataLoader(eval_data, batch_size=self.batch_size, collate_fn=eval_data.eval_batch_collate,
                                     shuffle=False)
        if verbose:
            eval_dataloader = tqdm(eval_dataloader, desc="Inference progress")

        self.model = self.model.cuda() if self.cuda else self.model.cpu()
        self.model.eval()

        predictions = []

        with torch.no_grad():
            for x_batch, lengths, condition in eval_dataloader:
                if self.cuda:
                    x_batch = x_batch.cuda()
                    condition = move_to_cuda(condition)
                else:
                    x_batch = x_batch.cpu()
                    condition = move_to_cpu(condition)

                prediction_scores_batch = self.model['encoder'](x_batch, lengths, condition)
                lengths = lengths.detach().cpu().tolist()

                predictions.extend(self._decode_prediction(prediction_scores_batch, lengths))

        return predictions
