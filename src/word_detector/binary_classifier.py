"""
Classify if a target word is present in a transcription sequence.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

def l2_logit_norm(prediction, target):
    """
    Apply L2 distance between two vectors.

    Is close to 0 for two similar vectors, close to 1 for different vectors
    """
    probabilities = torch.softmax(prediction, 0)
    val = torch.norm(probabilities - target, dim=None) / 1.414
    return val


def cosine_similarity(prediction, target):
    normed = torch.softmax(prediction, 0)
    similarity = (1 - torch.nn.CosineSimilarity(dim=0)(normed, target)) / 2
    return similarity


def argmax_selection(prediction, target):
    """
    Select normalized(prediction)[argmax(target)]
    0 for same vectors, 1 for totally different
    """
    normed = torch.softmax(prediction, 0)
    return 1 - normed[torch.argmax(target)]


def plot_metric(metric, prediction, target, decoder):
    """Plot the result of a metric."""
    _fig, ax = plt.subplots()
    predicted_labels = (
        decoder(i) if i < prediction.shape[0] - 4 else "" for i in range(prediction.shape[0])
    )
    normed = torch.softmax(prediction, 0)
    ax.plot(normed, label="Normed prediction")
    ax.scatter([torch.argmax(target).item()], [1], label="Target", marker="X")
    ax.set_xticks(range(prediction.shape[0]), predicted_labels)
    value = 1 - metric(prediction, target)
    ax.plot([0, normed.shape[0]], [value, value], label="1 - Metric value (1 = perfect)")
    plt.legend()
    plt.show()


def compute_path_matrix(prediction, target, metric, insertion_cost, deletion_cost):
    """Compute the alignment matrix of two matrices."""
    # Define the matrix
    path_matrix = torch.empty((prediction.shape[1], target.shape[1]))

    # Now run recursively
    for i, pred_column in enumerate(prediction[0]):
        for j, target_column in enumerate(target[0]):
            if i == 0 and j == 0:
                path_matrix[i, j] = 0
            elif i == 0:
                path_matrix[0, j] = j * insertion_cost
            elif j == 0:
                path_matrix[i, 0] = i * deletion_cost
            else:
                # plot_metric(metric, pred_column, target_column)
                path_matrix[i, j] = min(
                    path_matrix[i - 1, j - 1] + metric(pred_column, target_column),
                    path_matrix[i - 1, j] + deletion_cost,
                    path_matrix[i, j - 1] + insertion_cost
                )
    return path_matrix


def solve_path(prediction, target, path_matrix):
    """
    Find the matching path between a prediction, a target and a path matrix.

    For each step we minimize the cost.
    """
    line, col = prediction.shape[1] - 1, target.shape[1] - 1
    matching = []
    while line > 0 or col > 0:
        matching.append((line, col))
        directions = []
        if line > 0 and col > 0:
            directions.append((line - 1, col - 1))
        if line > 0:
            directions.append((line - 1, col))
        if col > 0:
            directions.append((line, col - 1))
        best_score = float("inf")
        dir_index = -1
        for i, direction in enumerate(directions):
            if path_matrix[direction[0]][direction[1]] < best_score:
                best_score = path_matrix[direction[0]][direction[1]]
                dir_index = i
        line, col = directions[dir_index]
    matching.reverse()
    return matching


def display_matrix_result(path_matrix, matching, prediction, target, decoder):
    """Display all the information resulting from a Bellman matching of matrices."""
    _fig, axis = plt.subplots()

    # Display the matrix
    axis.matshow(path_matrix.T, aspect="auto")

    # Set the labels for the axes
    axis.set_xlabel('Predicted String')
    # String for the x-axis
    predicted_labels = tuple(map(decoder, torch.argmax(prediction, -1)[0]))
    axis.set_xticks(
        [i for i, label in enumerate(predicted_labels) if label == ""],
        labels=[label for label in predicted_labels if label == ""]
    )
    axis.set_xticks(
        [i for i, label in enumerate(predicted_labels) if label not in ("[PAD]", "")],
        labels=[label for label in predicted_labels if label not in ("[PAD]", "")],
        minor=True
    )

    axis.set_ylabel('Target String')
    target_labels = tuple(map(decoder, torch.argmax(target, -1)[0]))
    axis.set_yticks(
        [i for i, label in enumerate(target_labels) if label == ""],
        labels=[label for label in target_labels if label == ""]
    )
    axis.set_yticks(
        [i for i, label in enumerate(target_labels) if label != ""],
        labels=[label for label in target_labels if label != ""],
        minor=True
    )
    # axis.yaxis.grid(which="major", color='k', linestyle='--')

    axis.grid(which="major", color="black")
    axis.grid(which="minor", linestyle="--")
    axis.plot(
        [val[0] for val in matching],
        [val[1] for val in matching],
        color="red"
    )
    plt.show()


def bellman_matching(prediction, target, insertion_cost=1.3, deletion_cost=3, metric=l2_logit_norm):
    """
    Match to sequences with Bellman's algorithm.

    :param prediction: Actual prediction
    :param target: Target list of values.
    :param float insertion_cost: Something was added in prediction.
    :param float deletion_cost: Something was missing in prediction.
    :param Callable metric: The metric to use.
    :return tuple(list, float): Best alignment [(prediction[i], target[j]), ...] for all elements, and its score
    """

    # Add padding: start matching on letters (do not penalize kids starting with insertions or long audio)
    padded_target = torch.zeros((target.shape[0], target.shape[1] + 1, target.shape[2]))
    padded_target[0, 1:] = target

    padded_prediction = torch.zeros((prediction.shape[0], prediction.shape[1] + 1, prediction.shape[2]))
    padded_prediction[0, 1:] = prediction

    path_matrix = compute_path_matrix(
        padded_prediction, padded_target, metric,
        insertion_cost,
        deletion_cost
    )
    # Now solve path, find candidate diagonal
    padded_matching = solve_path(padded_prediction, padded_target, path_matrix)

    short_matching = []
    for match in padded_matching:
        if match[0] == 0 or match[1] == 0:
            continue
        short_matching.append((match[0] - 1, match[1] - 1))
        if match[1] == padded_target.shape[1] - 1:
            break

    # display_matrix_result(path_matrix, padded_matching, padded_prediction, padded_target)
    # Initial padding should not reduce score
    score = path_matrix[padded_matching[-1]]

    return short_matching, score.item()


def get_alignment_score(prediction, target, weights):
    """
    Get a classification score, either 0, 1 or 2 from a prediction and a target.

    Both the prediction and the target should be logits.

    The result is 2 is no mistake, 1 if 1 mistake, 0 otherwise.
    """
    logits = torch.tensor(prediction)
    reduced_logits = logits[torch.argmax(logits, -1) != 58]
    reduced_logits = reduced_logits.unsqueeze(0)

    matching, _alignment_score = bellman_matching(
        reduced_logits,
        target,
        insertion_cost=weights[0],
        deletion_cost=weights[1],
        metric=l2_logit_norm
    )

    # Now from the matching count errors
    insertions = deletions = substitutions = 0
    np_matching = np.array(matching)

    for i, match in enumerate(np_matching[1:]):
        if np.all(match - np_matching[i] == [0, 1]):
            # Deletion occurred
            deletions += 1
        elif np.all(match - np_matching[i] == [1, 0]):
            # Insertion
            insertions += 1
        else:
            # Match probability, 1 == good match
            match_value = 1 - argmax_selection(reduced_logits[0, match[0]], target[0, match[1]])
            if match_value < weights[2]:
                substitutions += 1

    if insertions + deletions + substitutions == 0:
        return 1
    return 0
