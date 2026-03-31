import torch
import random
import numpy as np
import os
import glob
import warnings
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize


class DaliaActivityDataset(Dataset):
    """
    Dataset for DALIA Activity Classification
    9-class classification task
    Uses PPG signal only
    """

    def __init__(self, x_data, y_data, mode='train'):
        """
        Args:
            x_data: shape (N, 1, 400) - PPG only
            y_data: shape (N,) - Activity labels (0-8)
            mode: 'train' or 'test'
        """
        self.x = x_data
        self.y = y_data
        self.mode = mode
        self.sig_len = 400  # 8s * 50Hz

        print(f"Dataset initialized in {mode} mode with signal shape: {self.x.shape}, label shape: {self.y.shape}")
        print(f"  Using PPG only (1 channel)")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Input shape: (1, 400) - PPG only
        sig = self.x[idx]
        label = self.y[idx]

        # Ensure memory is writable and contiguous
        if isinstance(sig, np.ndarray):
            sig = torch.from_numpy(sig.copy()).float()
        elif not isinstance(sig, torch.Tensor):
            sig = torch.tensor(sig).float()

        if sig.ndim > 2:
            print(f"Warning: Signal has unexpected dimensions {sig.shape}. Reshaping...")
            sig = sig.reshape(sig.shape[0], -1)

        # Ensure correct length
        current_len = sig.shape[-1]
        if current_len != self.sig_len:
            print(f"Warning: Signal length mismatch! Expected {self.sig_len}, got {current_len}. Fixing...")
            if current_len > self.sig_len:
                # Truncate
                sig = sig[:, :self.sig_len]
            else:
                # Pad with zeros
                padding = torch.zeros(sig.shape[0], self.sig_len - current_len)
                sig = torch.cat([sig, padding], dim=1)

        # Handle NaN/Inf values
        if torch.isnan(sig).any() or torch.isinf(sig).any():
            sig = torch.nan_to_num(sig, nan=0.0, posinf=1.0, neginf=-1.0)

        # Data augmentation for training
        if self.mode == 'train':
            # Amplitude scaling
            if random.random() < 0.5:
                scale_factor = random.uniform(0.9, 1.1)
                sig = sig * scale_factor

            # Add Gaussian noise
            if random.random() < 0.5:
                noise = torch.randn_like(sig) * 0.05
                sig = sig + noise

            # Time shifting (circular shift)
            if random.random() < 0.3:
                shift = random.randint(-20, 20)  # shift by ±0.4s at 50Hz
                sig = torch.roll(sig, shift, dims=-1)

        # Label
        if isinstance(label, np.ndarray) or isinstance(label, np.integer):
            label = torch.tensor(label).long()

        return sig, label


def load_dalia_all_data(data_dir):
    """
    Load all DALIA data for k-fold cross validation
    Uses PPG only

    Args:
        data_dir: directory containing preprocessed data

    Returns:
        all_x, all_y: all data concatenated
    """
    all_files = glob.glob(os.path.join(data_dir, "*_x.npy"))
    if len(all_files) == 0:
        raise ValueError(f"No .npy files found in {data_dir}. Did you run preprocessing?")

    all_x_list, all_y_list = [], []

    # Extract all subject IDs
    subjects = sorted(
        list(set([os.path.basename(f).split('_')[0] for f in all_files])),
        key=lambda x: int(x[1:])
    )

    print(f"\nLoading all DALIA data from {len(subjects)} subjects...")

    for subj in subjects:
        x_path = os.path.join(data_dir, f"{subj}_x.npy")
        y_path = os.path.join(data_dir, f"{subj}_y.npy")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            print(f"Warning: Data for {subj} not found. Skipping...")
            continue

        x = np.load(x_path)
        y = np.load(y_path)

        all_x_list.append(x)
        all_y_list.append(y)

        print(f"  {subj}: {x.shape[0]} samples")

    all_x = np.concatenate(all_x_list, axis=0)
    all_y = np.concatenate(all_y_list, axis=0)

    print(f"\nTotal data loaded:")
    print(f"  Shape: {all_x.shape}")
    print(f"  Classes: {len(np.unique(all_y))}")
    print(f"  Class distribution: {np.bincount(all_y, minlength=9)}")

    return all_x, all_y


def load_dalia_loso(data_dir, test_subject_id):
    """
    Load DALIA data for Leave-One-Subject-Out validation
    Uses PPG only

    Args:
        data_dir: directory containing preprocessed data
        test_subject_id: e.g. 'S1', 'S15'

    Returns:
        train_x, train_y, test_x, test_y
    """
    all_files = glob.glob(os.path.join(data_dir, "*_x.npy"))
    if len(all_files) == 0:
        raise ValueError(f"No .npy files found in {data_dir}. Did you run preprocessing?")

    train_x_list, train_y_list = [], []
    test_x, test_y = None, None

    # Extract all subject IDs
    subjects = sorted(
        list(set([os.path.basename(f).split('_')[0] for f in all_files])),
        key=lambda x: int(x[1:])
    )

    print(f"\nLeave-One-Subject-Out Setup:")
    print(f"  Test Subject: {test_subject_id}")
    print(f"  Train Subjects: {[s for s in subjects if s != test_subject_id]}")

    for subj in subjects:
        x_path = os.path.join(data_dir, f"{subj}_x.npy")
        y_path = os.path.join(data_dir, f"{subj}_y.npy")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            print(f"Warning: Data for {subj} not found. Skipping...")
            continue

        x = np.load(x_path)
        y = np.load(y_path)

        if subj == test_subject_id:
            test_x = x
            test_y = y
        else:
            train_x_list.append(x)
            train_y_list.append(y)

    if test_x is None:
        raise ValueError(f"Test subject {test_subject_id} not found in data.")

    train_x = np.concatenate(train_x_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)

    print(f"\nData loaded:")
    print(f"  Train: {train_x.shape}, {len(np.unique(train_y))} classes")
    print(f"  Test:  {test_x.shape}, {len(np.unique(test_y))} classes")
    print(f"  Train class distribution: {np.bincount(train_y, minlength=9)}")
    print(f"  Test class distribution:  {np.bincount(test_y, minlength=9)}")

    return train_x, train_y, test_x, test_y


def calculate_metrics(preds, targets, probs=None):
    """
    Calculate classification metrics including AUROC

    Args:
        preds: predicted labels (numpy array)
        targets: ground truth labels (numpy array)
        probs: predicted probabilities (numpy array, shape: [N, num_classes]), optional

    Returns:
        dict with accuracy, f1 score, and AUROC metrics (if probs provided)
    """
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')

    metrics = {
        'accuracy': acc,
        'f1': f1
    }

    # Calculate AUROC if probabilities are provided
    if probs is not None:
        # Check if we have enough classes for AUROC calculation
        n_classes_present = len(np.unique(targets))

        if n_classes_present < 2:
            # AUROC cannot be calculated with only one class
            metrics['auroc_macro'] = float('nan')
            metrics['auroc_weighted'] = float('nan')
            metrics['auroc_per_class'] = [float('nan')] * probs.shape[1]
        else:
            try:
                # Get number of classes
                num_classes = probs.shape[1]

                # Binarize the labels for one-vs-rest AUROC
                targets_bin = label_binarize(targets, classes=np.arange(num_classes))

                # Handle edge case: if only one class present, label_binarize returns 1D array
                if targets_bin.ndim == 1:
                    targets_bin = targets_bin.reshape(-1, 1)

                # Calculate AUROC for each class (one-vs-rest)
                auroc_per_class = []
                for i in range(num_classes):
                    if i in targets:
                        # Check if this class has both positive and negative samples
                        class_targets = targets_bin[:, i] if targets_bin.ndim > 1 else targets_bin
                        if len(np.unique(class_targets)) > 1:
                            try:
                                with warnings.catch_warnings():
                                    warnings.filterwarnings('ignore', category=UserWarning)
                                    auroc = roc_auc_score(
                                        targets_bin[:, i] if targets_bin.ndim > 1 else targets_bin,
                                        probs[:, i]
                                    )
                                auroc_per_class.append(auroc)
                            except Exception:
                                auroc_per_class.append(float('nan'))
                        else:
                            auroc_per_class.append(float('nan'))
                    else:
                        auroc_per_class.append(float('nan'))

                # Macro-average AUROC (average across all classes)
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning)
                        auroc_macro = roc_auc_score(
                            targets_bin,
                            probs,
                            average='macro',
                            multi_class='ovr'
                        )
                except Exception:
                    auroc_macro = float('nan')

                # Weighted-average AUROC
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning)
                        auroc_weighted = roc_auc_score(
                            targets_bin,
                            probs,
                            average='weighted',
                            multi_class='ovr'
                        )
                except Exception:
                    auroc_weighted = float('nan')

                metrics['auroc_macro'] = auroc_macro
                metrics['auroc_weighted'] = auroc_weighted
                metrics['auroc_per_class'] = auroc_per_class

            except Exception as e:
                # Fallback to NaN if any unexpected error occurs
                metrics['auroc_macro'] = float('nan')
                metrics['auroc_weighted'] = float('nan')
                metrics['auroc_per_class'] = [float('nan')] * probs.shape[1]

    return metrics