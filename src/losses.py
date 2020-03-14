# Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
import torch

import numpy as np
from scipy.sparse import coo_matrix


def endpoint_error(estimate, ground_truth):
    """Computes the average end-point error of the optical flow estimates."""
    error = torch.norm(
        estimate - ground_truth[:, :2, :, :], 2, 1, keepdim=False)
    if ground_truth.size(1) == 3:
        mask = (ground_truth[:, 2, :, :] > 0).float()
    else:
        mask = torch.ones_like(error)
    epe = error * mask
    epe = torch.sum(epe, (1, 2)) / torch.sum(mask, (1, 2))
    return epe.mean().reshape(1)


def outlier_rate(estimate, ground_truth):
    """Computes the 3-pixel outlier rate of the optical flow estimates."""
    error = torch.norm(
        estimate - ground_truth[:, :2, :, :], 2, 1, keepdim=False)
    if ground_truth.size(1) == 3:
        mask = (ground_truth[:, 2, :, :] > 0).float()
    else:
        mask = torch.ones_like(error)
    gt_magnitude = torch.norm(ground_truth[:, :2, :, :], 2, 1, keepdim=False)
    outliers = mask * (error > 3.0).float() * (
        (error / gt_magnitude) > 0.05).float()
    outliers = torch.sum(outliers, (1, 2)) * 100. / torch.sum(mask, (1, 2))
    return outliers.mean().reshape(1)


class CrossEntropySegmentationCalculator(object):
    """Calculates cross entropy loss for semantic segmentation."""

    def __init__(self):
        self.loss = torch.nn.CrossEntropyLoss(
            weight=None, reduction='mean', ignore_index=-1)

    def __call__(self, estimate, ground_truth):
        cross_entropy = self.loss(estimate, ground_truth[:, 0])
        return cross_entropy.mean().reshape(1)


class MiouMetric(object):
    """Mean intersection over union metric for semantic segmentation."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.confusion_matrix = 0

    def update(self, predictions, labels):
        """Updates the loss metric."""
        predictions = torch.argmax(predictions, dim=1)
        labels = labels.reshape([-1]).cpu().numpy()
        predictions = predictions.reshape([-1]).cpu().numpy()
        valid_labels = (labels >= 0)
        self.confusion_matrix += _compute_confusion_matrix(
            predictions[valid_labels], labels[valid_labels], self.num_classes)

    def get(self):
        """Gets the current evaluation result."""
        sum_rows = np.sum(self.confusion_matrix, 0)
        sum_colums = np.sum(self.confusion_matrix, 1)
        diagonal_entries = np.diag(self.confusion_matrix)
        denominator = sum_rows + sum_colums - diagonal_entries

        valid_classes = (denominator != 0)
        num_valid_classes = np.sum(valid_classes)
        denominator += (1 - valid_classes)
        iou = diagonal_entries / denominator
        if num_valid_classes == 0:
            return float('nan')
        return np.sum(iou) / num_valid_classes


def _compute_confusion_matrix(predictions, labels, num_classes):
    if np.min(labels) < 0 or np.max(labels) >= num_classes:
        raise Exception("Labels out of bound.")

    if np.min(predictions) < 0 or np.max(predictions) >= num_classes:
        raise Exception("Predictions out of bound.")

    # Idea borrowed from tensorflow implementation.
    values = np.ones(predictions.shape)
    confusion_matrix = coo_matrix((values, (labels, predictions)),
                                  shape=(num_classes, num_classes)).toarray()
    return confusion_matrix
