# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch
import pandas as pd
import numpy as np
import ast
import joblib
import os
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils import resample


def get_data_for_ml(df, dict_embeddings, case_name, label, level="patient"):
    """
    Extract features and label starting from the dictionary

    Args:
        df (pandas.DataFrame): dataframe containing user id, etc.
        dict_embeddings (dictionary): dictionary containing extracted embeddings
        case_name (string): column name for user id in dataframe
        label (string): label to extract
        level (string): patient, averages value of segments for a user

    Returns:
        X (np.array): feature array
        y (np.array): label array
        keys (list): test keys

    """
    y = []
    if level == "patient":
        df = df.drop_duplicates(subset=[case_name])

    for key in dict_embeddings.keys():
        if level == "patient":
            y.append(df[df.loc[:, case_name] == key].loc[:, label].values[0])
        else:
            y.append(df[df.loc[:, case_name] == key].loc[:, label].values)
    X = np.vstack([k.cpu().detach().numpy() if type(k) == torch.Tensor else k for k in dict_embeddings.values()])
    y = np.hstack(y)
    return X, y, list(dict_embeddings.keys())


def get_data_for_ml_from_df(df, dict_embeddings, case_name, label, level="patient"):
    """
    Extract features and label starting from the dataframe

    Args:
        df (pandas.DataFrame): dataframe containing user id, etc.
        dict_embeddings (dictionary): dictionary containing extracted embeddings
        case_name (string): column name for user id in dataframe
        label (string): label to extract
        level (string): patient, averages value of segments for a user

    Returns:
        X (np.array): feature array
        y (np.array): label array
        keys (list): test keys
    """
    X = []
    y = []
    df = df.drop_duplicates(subset=[case_name])
    filenames = df[case_name].values
    for f in filenames:
        if f in dict_embeddings.keys():
            if level == "patient":
                y.append(df[df.loc[:, case_name] == f].loc[:, label].values[0])
            else:
                y.append(df[df.loc[:, case_name] == f].loc[:, label].values)
            X.append([k.cpu().detach().numpy() if type(k) == torch.Tensor else k for k in dict_embeddings[f]])
    X = np.vstack(X)
    return X, np.array(y), filenames


def extract_labels(y, label, binarize_val=None):
    """
    The raw labels are converted to categorical for classification

    Args:
        y (np.array): label array in raw form
        label (string) :label name
        binarize_val: Use the median to binarize the label

    Returns:
        y (np.array): label array ready for trianing/eval
    """

    if label == "age":
        y = np.where(y > 50, 1, 0)

    if label == "sex":
        y = np.where(y == "M", 1, 0)

    if label in ['bmi', 'es', 'cr', 'TMD']:
        y = np.where(y > binarize_val, 1, 0)

    if label == "icu_days":
        y = np.where(y > 0, 1, 0)

    if label == "death_inhosp":
        y = y

    if label == "optype":
        dict_label = {'Colorectal': 0,
                      'Biliary/Pancreas': 1,
                      'Stomach': 2,
                      'Others': 3,
                      'Major resection': 4,
                      'Minor resection': 5,
                      'Breast': 6,
                      'Transplantation': 7,
                      'Thyroid': 8,
                      'Hepatic': 9,
                      'Vascular': 10}
        y = [dict_label[op] for op in y]

    if label == "AHI":
        y = np.where(y > 0, 1, 0)

    if label == "Hypertension":
        y = np.where(y == "Normal", 0, 1)

    if label == "Diabetes" or label == "cerebrovascular disease" or label == "cerebral infarction":
        y = np.where(y == "0", 0, 1)

    if label == "valence" or label == "arousal":
        y = np.where(y <= 5, 1, 0)

    if label == "affect":
        y = y

    if label == "activity":
        y = y

    if label == "nsrr_current_smoker" or label == "nsrr_ever_smoker":
        y = np.where(y == "yes", 1, 0)

    if label == "sds":
        y = np.where(y > 49, 1, 0)

    if label == "DOD":
        y = np.where(pd.notna(y), 1, 0)

    if label == "stdyvis":
        y = np.where(y == 3, 1, 0)

    if label == "afib":
        y = np.where(y == "af", 1, 0)

    return y


def bootstrap_metric_confidence_interval(y_test, y_pred, metric_func, num_bootstrap_samples=500, confidence_level=0.95):
    bootstrapped_metrics = []

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Bootstrap sampling
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        indices = np.random.choice(range(len(y_test)), size=len(y_test), replace=True)
        y_test_sample = y_test[indices]
        y_pred_sample = y_pred[indices]

        # Calculate the metric for the resampled data
        metric_value = metric_func(y_test_sample, y_pred_sample)
        bootstrapped_metrics.append(metric_value)

    # Calculate the confidence interval
    lower_bound = np.percentile(bootstrapped_metrics, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_metrics, (1 + confidence_level) / 2 * 100)

    return lower_bound, upper_bound, bootstrapped_metrics


def sanitize(arr):
    """
    Convert an list/array from a string to a float array
    """
    parsed_list = ast.literal_eval(arr)
    return np.array(parsed_list, dtype=float)


def load_model(model, filepath):
    """
    Load a PyTorch model from a specified file path.

    Args:
    model (torch.nn.Module): The PyTorch model instance to load the state dictionary into.
    filepath (str): The path from which the model will be loaded.

    Returns:
    model (torch.nn.Module): The model with the loaded state dictionary.
    """
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {filepath}")
    return model


def batch_load_signals(path, case, segments):
    """
    Load ppg segments in batches
    """
    batch_signal = []
    for s in segments:
        batch_signal.append(joblib.load(os.path.join(path, case, str(s))))
    return np.vstack(batch_signal)


def load_model_without_module_prefix(model, checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Create a new state_dict with the `module.` prefix removed
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_key = k[7:]  # Remove `module.` prefix
        else:
            new_key = k
        new_state_dict[new_key] = v

    # Load the new state_dict into the model
    model.load_state_dict(new_state_dict)

    return model