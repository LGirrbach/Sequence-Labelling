import numpy as np

from typing import Any
from typing import List
from collections import Counter
from constants import SequenceData
from models.neural_labeller import NeuralLabeller


def _get_voted_prediction(predictions: List[Any], scores: List[float], max_score: bool = False):
    votes = Counter(predictions)

    max_votes = max(votes.values())
    candidates_indices = [
        idx for idx, instance_prediction in enumerate(predictions)
        if votes[instance_prediction] == max_votes
    ]

    candidate_scores = [scores[idx] for idx in candidates_indices]
    best_idx = np.argmax(candidate_scores).item() if max_score else np.argmin(candidate_scores).item()
    best_candidate_idx = candidates_indices[best_idx]
    best_candidate = predictions[best_candidate_idx]

    return best_candidate


def get_sequence_votes(predictions: List[SequenceData], model_scores: List[float],
                       max_score: bool = False) -> SequenceData:
    voted_predictions = []

    for instance_predictions in zip(*predictions):
        if all(isinstance(instance_prediction, list) for instance_prediction in instance_predictions):
            instance_predictions = [tuple(instance_prediction) for instance_prediction in instance_predictions]
        best_candidate = _get_voted_prediction(instance_predictions, model_scores, max_score)
        voted_predictions.append(best_candidate)

    return voted_predictions


def get_position_votes(predictions: List[SequenceData], model_scores: List[float],
                       max_score: bool = False) -> SequenceData:
    voted_predictions = []

    for instance_predictions in zip(*predictions):
        instance_predictions = [tuple(instance_prediction) for instance_prediction in instance_predictions]
        instance_prediction = []

        for position_predictions in zip(*instance_predictions):
            position_predictions = list(position_predictions)
            best_prediction = _get_voted_prediction(position_predictions, model_scores, max_score)
            instance_prediction.append(best_prediction)

        voted_predictions.append(instance_prediction)

    return voted_predictions


def ensemble_predict(models: List[NeuralLabeller], *data_args, max_score: bool = False, **data_kwargs):
    model_predictions = [
        model.predict(*data_args, **data_kwargs) for model in models
    ]
    model_scores = [model.best_dev_score for model in models]

    postprocessed_predictions = [
        [model.postprocess_prediction(prediction) for prediction in predictions]
        for predictions, model in zip(model_predictions, models)
    ]

    sequence_voted_predictions = get_sequence_votes(postprocessed_predictions, model_scores, max_score)
    position_voted_predictions = get_position_votes(model_predictions, model_scores, max_score)

    position_voted_predictions = [
        models[0].postprocess_prediction(prediction) for prediction in position_voted_predictions
    ]

    return {
        'sequence_votes': sequence_voted_predictions,
        'position_votes': position_voted_predictions
    }
