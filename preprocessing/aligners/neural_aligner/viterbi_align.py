import numpy as np

# Traceback codes:
DIAGONAL = 1
HORIZONTAL = 2
VERTICAL = 3


def get_best_path(scores):
    # Keep track of the best path probabilities
    alpha = np.zeros_like(scores)
    # Keep track of paths
    traceback = np.zeros_like(scores)

    # Initialise upper left corner (has to be aligned)
    alpha[0, 0] = scores[0, 0]
    traceback[0][0] = DIAGONAL

    # Handle cases where less than 3 predecessors exist
    for i in range(1, alpha.shape[0]):
        alpha[i, 0] = scores[i, 0] + alpha[i - 1, 0]
        traceback[i, 0] = VERTICAL

    for i in range(1, alpha.shape[1]):
        alpha[0, i] = scores[0, i] + alpha[0, i - 1]
        traceback[0, i] = HORIZONTAL

    # Find best path by viterbi
    for i in range(1, alpha.shape[0]):
        for j in range(1, alpha.shape[1]):
            prev_scores = [alpha[i - 1, j - 1], alpha[i, j - 1], alpha[i - 1, j]]
            best_option = np.argmax(prev_scores)
            alpha[i, j] = scores[i, j] + prev_scores[best_option]
            traceback[i, j] = best_option + 1

    # For the alignment, we only need the path, not it's probability
    return traceback


def make_alignment(source, target, traceback):
    source_alignment = []
    target_alignment = []

    current_source_chunk = []
    current_target_chunk = []
    i, j = traceback.shape[0] - 1, traceback.shape[1] - 1

    while i >= 0 or j >= 0:
        op = traceback[i, j]
        if op == DIAGONAL:
            current_source_chunk.append(source[i])
            current_target_chunk.append(target[j])

            source_alignment.append(current_source_chunk[::-1])
            target_alignment.append(current_target_chunk[::-1])

            current_source_chunk = []
            current_target_chunk = []
            i, j = i - 1, j - 1

        elif op == HORIZONTAL:
            current_target_chunk.append(target[j])
            i, j = i, j - 1

        elif op == VERTICAL:
            current_source_chunk.append(source[i])
            i, j = i - 1, j

    source_alignment = source_alignment[::-1]
    target_alignment = target_alignment[::-1]

    return source_alignment, target_alignment


def viterbi_align(source, target, scores):
    traceback = get_best_path(scores)
    return make_alignment(source, target, traceback)
