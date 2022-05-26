import random
import numpy as np

from typing import List, Tuple

BLANK = "#"
DIAGONAL = 0
HORIZONTAL = 1
VERTICAL = 2


def convert_indices_to_str(s1: str, s2: str, index_alignments: List[Tuple[int, int]]):
    alignment = []
    last_s1_idx, last_s2_idx = None, None
    for s1_idx, s2_idx in reversed(index_alignments):
        s1_idx, s2_idx = s1_idx - 1, s2_idx - 1
        s1_aligned_char = BLANK if (s1_idx == last_s1_idx or s1_idx < 0) else s1[s1_idx]
        s2_aligned_char = BLANK if (s2_idx == last_s2_idx or s2_idx < 0) else s2[s2_idx]
        last_s1_idx, last_s2_idx = s1_idx, s2_idx

        alignment.append((s1_aligned_char, s2_aligned_char))

    return alignment


def score_alignment(alignment: List[Tuple[str, str]]):
    """
    Scores alignments by the sum of the squared lengths of contiguous alignments.
    This favours alignment of longer contiguous subsequences, e.g.
    --erden-    vs.    --e-rden-
    geerde-t           geerde-t
    (the first one is preferred)
    """
    score = 0
    segment_length = 0

    for aligned_chars in alignment:
        if BLANK not in aligned_chars:
            segment_length += 1
        else:
            score += segment_length ** 2
            segment_length = 0

    # Add score for final segment
    score += segment_length ** 2

    return score


def score_substitution(item1, item2):
    return 1 if item1 == item2 else -10000


def reconstruct_alignment(s1: str, s2: str, traceback: List[List[List[int]]], max_candidates=1e7,
                          candidate_scorer=score_alignment):
    best_alignment = None
    best_score = -np.inf
    candidate_counter = 0

    queue = [(len(traceback)-1, len(traceback[0])-1, [])]

    while len(queue) > 0 and candidate_counter < max_candidates:
        if len(queue) > max_candidates:
            queue = queue[-max_candidates:]
        i, j, candidate = queue.pop()

        # Check if candidate has been fully reconstructed
        if i == 0 and j == 0:
            candidate = convert_indices_to_str(s1, s2, candidate)
            candidate_score = candidate_scorer(candidate)
            candidate_counter += 1

            if candidate_score > best_score:
                best_alignment = candidate.copy()
                best_score = candidate_score

        else:
            predecessor_directions = traceback[i][j]
            random.shuffle(predecessor_directions)
            for direction in predecessor_directions:
                if direction == DIAGONAL:
                    queue.append((i-1, j-1, candidate + [(i, j)]))

                elif direction == HORIZONTAL:
                    queue.append((i, j-1, candidate + [(i, j)]))

                elif direction == VERTICAL:
                    queue.append((i-1, j, candidate + [(i, j)]))

                else:
                    raise ValueError(f"Unknown direction: {direction}")

    return best_alignment


def align(s1: str, s2: str, scorer=score_substitution, gap_cost: int = -1, candidate_scorer=score_alignment):
    """Align a new form to the previously computed MSA alignment of forms."""
    s1_len, s2_len = len(s1), len(s2)

    # Initialise score matrix
    score_matrix = np.zeros(shape=(s1_len+1, s2_len+1))
    # Initialise traceback (=backpointer) matrix
    # We want to reconstruct all alignments with max. score, so we keep all relevant backpointers
    traceback = [[[] for _ in range(s2_len+1)] for _ in range(s1_len+1)]

    # Initialise deletion of MSA columns
    for k in range(1, s2_len+1):
        score_matrix[0, k] = score_matrix[0, k-1] + gap_cost
        traceback[0][k].append(HORIZONTAL)

    # Initialise deletion of form characters
    for k in range(1, s1_len+1):
        score_matrix[k, 0] = score_matrix[k-1, 0] + gap_cost
        traceback[k][0].append(VERTICAL)

    # Calculate edit distance
    for i in range(1, s1_len+1):
        for j in range(1, s2_len+1):
            diagonal_score = score_matrix[i-1, j-1] + scorer(s1[i-1], s2[j-1])
            # Gap in s2
            horizontal_score = score_matrix[i, j-1] + gap_cost
            # Gap in s2
            vertical_score = score_matrix[i-1, j] + gap_cost

            # Find best scores and add operations to traceback
            scores = [diagonal_score, horizontal_score, vertical_score]
            best_score = max(scores)
            score_matrix[i, j] = best_score
            for k, score in enumerate(scores):
                if score == best_score:
                    traceback[i][j].append(k)

    # Reconstruct alignment with maximum score
    best_alignment = reconstruct_alignment(s1, s2, traceback, candidate_scorer=candidate_scorer)

    return best_alignment
