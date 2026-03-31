import logging
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import numpy as np


def get_logger(
        logpath, filepath, package_files=[], displaying=True, saving=True, debug=False
):
    """
    Setup logger for training experiments
    Identical to VTac project logger
    """
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="w")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


class Dataset_train(Dataset):
    """
    WESAD Dataset for training/validation/testing
    Adapted for SOTA preprocessing format (50Hz, 3000 samples/window)

    For WESAD SOTA:
    - target_len: 3000 (60 seconds * 50 Hz) - full window
    - end_idx: 3000 (use full window from end)

    Changes from original:
    - Updated for 50Hz sampling rate (was 125Hz)
    - Reduced window size from 7500 to 3000
    - Adjusted data augmentation parameters for lower sampling rate
    """

    def __init__(self, signal_train, y_train, mode='train', target_len=3000, end_idx=3000):
        self.strain = signal_train
        self.ytrain = y_train
        self.mode = mode
        self.target_len = target_len  # 3000 for 60-second windows at 50Hz
        self.end_idx = end_idx  # 3000 to use full window

        # Ensure signals are 2D (add channel dimension if needed)
        if self.strain.ndim == 1:
            self.strain = self.strain.unsqueeze(1)
        elif self.strain.ndim == 2:
            # Already correct shape (N, T) - will add channel dim in __getitem__
            pass

    def __len__(self):
        return len(self.ytrain)

    def __getitem__(self, index):
        full_sig = self.strain[index]

        # Add channel dimension if needed
        if full_sig.ndim == 1:
            full_sig = full_sig.unsqueeze(0)  # (1, T)

        start = self.end_idx - self.target_len
        end = self.end_idx

        # Data augmentation: random time shift (only during training)
        if self.mode == 'train':
            # Adjusted jitter for 50Hz (±0.8 seconds at 50Hz = ±40 samples)
            jitter = random.randint(-40, 40)  # Scaled down from -100 to -40
            if start + jitter >= 0 and end + jitter <= full_sig.shape[-1]:
                start += jitter
                end += jitter

        sig = full_sig[..., start:end]

        # Data augmentation: noise injection (only during training)
        if self.mode == 'train':
            if random.random() < 0.3:  # 30% probability
                noise = torch.randn_like(sig) * 0.05  # Small noise for normalized signals
                sig = sig + noise

            # Data augmentation: amplitude scaling (only during training)
            if random.random() < 0.3:  # 30% probability
                scale_factor = random.uniform(0.95, 1.05)  # Subtle scaling
                sig = sig * scale_factor

        return sig, self.ytrain[index]


class Dataset_multiclass(Dataset):
    """
    WESAD Dataset for multiclass classification (4 classes)

    For WESAD SOTA multiclass:
    - target_len: 3000 (60 seconds * 50 Hz)
    - num_classes: 4 (baseline, stress, amusement, meditation)
    """

    def __init__(self, signal_train, y_train, mode='train', target_len=3000, end_idx=3000):
        self.strain = signal_train
        self.ytrain = y_train
        self.mode = mode
        self.target_len = target_len
        self.end_idx = end_idx

        if self.strain.ndim == 1:
            self.strain = self.strain.unsqueeze(1)

    def __len__(self):
        return len(self.ytrain)

    def __getitem__(self, index):
        full_sig = self.strain[index]

        if full_sig.ndim == 1:
            full_sig = full_sig.unsqueeze(0)

        start = self.end_idx - self.target_len
        end = self.end_idx

        if self.mode == 'train':
            jitter = random.randint(-40, 40)
            if start + jitter >= 0 and end + jitter <= full_sig.shape[-1]:
                start += jitter
                end += jitter

        sig = full_sig[..., start:end]

        if self.mode == 'train':
            if random.random() < 0.3:
                noise = torch.randn_like(sig) * 0.05
                sig = sig + noise

            if random.random() < 0.3:
                scale_factor = random.uniform(0.95, 1.05)
                sig = sig * scale_factor

        return sig, self.ytrain[index]


def train_model(batch, model, loss_fn, device, weight=1.0):
    """
    Training step for WESAD model
    Supports both binary and multiclass classification

    Args:
        batch: (signal, labels) tuple
        model: Neural network model
        loss_fn: Loss function (BCEWithLogitsLoss for binary, CrossEntropyLoss for multiclass)
        device: torch device
        weight: Loss weight (not used in current setup but kept for compatibility)

    Returns:
        loss: Training loss
        predictions: Model predictions
        labels: Ground truth labels
    """
    signal_train, y_train = batch

    signal_train = signal_train.to(device)
    y_train = y_train.to(device)

    Y_train_prediction = model(signal_train)

    # Handle different loss function types
    if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
        # Binary classification
        y_train = y_train.float().view(-1, 1)
        loss = loss_fn(Y_train_prediction, y_train)
    else:
        # Multiclass classification
        y_train = y_train.long()
        loss = loss_fn(Y_train_prediction, y_train)

    return loss, Y_train_prediction, y_train


def eval_model(batch, model, loss_fn, device):
    """
    Evaluation step for WESAD model
    Supports both binary and multiclass classification

    Args:
        batch: (signal, labels) tuple
        model: Neural network model
        loss_fn: Loss function
        device: torch device

    Returns:
        loss: Evaluation loss
        predictions: Model predictions
        labels: Ground truth labels
    """
    signal_train, y_train = batch

    signal_train = signal_train.to(device)
    y_train = y_train.to(device)

    Y_train_prediction = model(signal_train)

    # Handle different loss function types
    if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
        # Binary classification
        y_train = y_train.float().view(-1, 1)
        loss = loss_fn(Y_train_prediction, y_train)
    else:
        # Multiclass classification
        y_train = y_train.long()
        loss = loss_fn(Y_train_prediction, y_train)

    return loss, Y_train_prediction, y_train


def evaluation(Y_eval_prediction, y_test, TP, FP, TN, FN):
    """
    Calculate confusion matrix metrics during training (binary classification)

    Args:
        Y_eval_prediction: Model predictions (logits)
        y_test: Ground truth labels
        TP, FP, TN, FN: Running counts of confusion matrix

    Returns:
        Updated TP, FP, TN, FN counts
    """
    # Convert logits to binary predictions (threshold at 0)
    pre = (Y_eval_prediction >= 0).int()

    for i, j in zip(pre, y_test):
        if i.item() == 1 and j.item() == 1:  # True Positive
            TP += 1
        if i.item() == 1 and j.item() == 0:  # False Positive
            FP += 1
        if i.item() == 0 and j.item() == 0:  # True Negative
            TN += 1
        if i.item() == 0 and j.item() == 1:  # False Negative
            FN += 1

    return TP, FP, TN, FN


def evaluation_test(Y_eval_prediction, y_test, types_TP, types_FP, types_TN, types_FN):
    """
    Calculate confusion matrix metrics during testing (binary classification)
    Same as evaluation() but with different variable names for clarity

    Args:
        Y_eval_prediction: Model predictions (logits)
        y_test: Ground truth labels
        types_TP, types_FP, types_TN, types_FN: Running counts

    Returns:
        Updated confusion matrix counts
    """
    pre = (Y_eval_prediction >= 0).int()

    for i, j in zip(pre, y_test):
        if i.item() == 1 and j.item() == 1:  # True Positive
            types_TP += 1
        if i.item() == 1 and j.item() == 0:  # False Positive
            types_FP += 1
        if i.item() == 0 and j.item() == 0:  # True Negative
            types_TN += 1
        if i.item() == 0 and j.item() == 1:  # False Negative
            types_FN += 1

    return types_TP, types_FP, types_TN, types_FN


def evaluation_multiclass(y_pred, y_true, confusion_matrix):
    """
    Calculate confusion matrix for multiclass classification

    Args:
        y_pred: Model predictions (logits or class indices)
        y_true: Ground truth labels
        confusion_matrix: numpy array (num_classes, num_classes)

    Returns:
        Updated confusion matrix
    """
    # Convert logits to class predictions if needed
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        y_pred = torch.argmax(y_pred, dim=-1)

    for pred, true in zip(y_pred, y_true):
        confusion_matrix[true.item()][pred.item()] += 1

    return confusion_matrix


def evaluate_raise_threshold(
        prediction, groundtruth, types_TP, types_FP, types_TN, types_FN, threshold=0.5
):
    """
    Evaluate with custom probability threshold (binary classification)
    Useful for adjusting precision/recall tradeoff

    Args:
        prediction: Model prediction (logit)
        groundtruth: Ground truth label
        types_TP, types_FP, types_TN, types_FN: Running counts
        threshold: Decision threshold (default 0.5 for probabilities)

    Returns:
        Updated confusion matrix counts
    """
    # Convert logit to probability
    prediction = torch.sigmoid(prediction)

    pre = 1 if prediction >= threshold else 0

    if pre == 1 and groundtruth == 1:
        types_TP += 1
    elif pre == 1 and groundtruth == 0:
        types_FP += 1
    elif pre == 0 and groundtruth == 1:
        types_FN += 1
    elif pre == 0 and groundtruth == 0:
        types_TN += 1

    return types_TP, types_FP, types_TN, types_FN


def evaluate_rule_based(rule_based_results, y_test):
    """
    Evaluate rule-based baseline methods

    Args:
        rule_based_results: Binary predictions from rule-based method
        y_test: Ground truth labels

    Returns:
        Sensitivity (TPR), Specificity (TNR), Weighted Score, Accuracy
    """
    TP = FP = TN = FN = 0

    for i, j in zip(rule_based_results, y_test):
        if i.item() == 1 and j.item() == 1:
            TP += 1
        if i.item() == 1 and j.item() == 0:
            FP += 1
        if i.item() == 0 and j.item() == 0:
            TN += 1
        if i.item() == 0 and j.item() == 1:
            FN += 1

    # Calculate metrics
    sensitivity = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = 100 * TN / (TN + FP) if (TN + FP) > 0 else 0
    weighted_score = 100 * (TP + TN) / (TP + TN + FP + 5 * FN) if (TP + TN + FP + FN) > 0 else 0
    accuracy = 100 * (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    return sensitivity, specificity, weighted_score, accuracy