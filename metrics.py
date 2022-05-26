import numpy as np
import editdistance

from constants import SequenceData


def sequence_accuracy(predictions: SequenceData, targets: SequenceData):
    assert len(predictions) == len(targets)
    num_correct, num_total = 0, len(targets)

    for predicted_labels, true_labels in zip(predictions, targets):
        if predicted_labels == true_labels:
            num_correct += 1

    return num_correct / num_total


def token_accuracy(predictions: SequenceData, targets: SequenceData) -> float:
    assert len(targets) == len(predictions)
    num_correct, total = 0, 0
    for predicted_labels, true_labels in zip(predictions, targets):
        if len(predicted_labels) != len(true_labels):
            total += len(true_labels)
            continue

        for predicted_label, true_label in zip(predicted_labels, true_labels):
            num_correct += (predicted_label == true_label)
            total += 1
    if total == 0:
        return 0.0
    return num_correct / total


def edit_distance(predictions: SequenceData, targets: SequenceData) -> float:
    assert len(predictions) == len(targets)
    distances = [
        editdistance.distance(predicted_labels, true_labels) for predicted_labels, true_labels
        in zip(predictions, targets)
    ]

    if len(distances) == 0:
        return 0.0

    return np.mean(distances).item()


def normalised_edit_distance(predictions: SequenceData, targets: SequenceData) -> float:
    assert len(predictions) == len(targets)
    distances = [
        editdistance.distance(predicted_labels, true_labels) / len(true_labels)
        for predicted_labels, true_labels
        in zip(predictions, targets)
    ]

    if len(distances) == 0:
        return 0.0

    return np.mean(distances).item()
