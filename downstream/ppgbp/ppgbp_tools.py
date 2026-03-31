"""
PPGBP工具类 - K折交叉验证版本
支持4个任务：Systolic BP (sbp), Diastolic BP (dbp), Hypertension (hyper), Heart Rate (hr)

主要功能：
1. 数据加载：从预处理的.npy文件加载K折数据
2. 简单数据增强：只使用高斯噪声
3. 指标计算：
   - 回归: MAE, RMSE, Correlation
   - 分类: Accuracy, F1, AUROC

默认采样率: 50Hz
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import pearsonr


class PPGBPDataset(Dataset):
    """简化的PPGBP数据集类，只使用基础的高斯噪声增强"""

    def __init__(self, x_data, y_data, task_name='sbp', mode='train', noise_std=0.0):
        self.x = x_data
        self.y = y_data
        self.task_name = task_name.lower()
        self.mode = mode
        self.noise_std = noise_std

        self.is_classification = (self.task_name == 'hyper')

        if self.is_classification:
            unique_labels = np.unique(self.y)
            n_classes = len(unique_labels)
            min_label = unique_labels.min()
            max_label = unique_labels.max()

            print(f"  [{mode}] Raw labels found: {unique_labels}")

            # ========== 处理多类别情况 ==========
            if n_classes > 2:
                print(f"  [{mode}] ⚠️  Found {n_classes} classes, merging to binary...")
                print(f"  [{mode}] Strategy: Class {min_label} → 0, Others → 1")

                # 第一个类别(最小值)映射为0，其他所有类别映射为1
                self.y = (self.y > min_label).astype(np.int64)

                # 显示转换结果
                new_unique = np.unique(self.y)
                print(f"  [{mode}] After merging: {new_unique}")

                # 显示类别分布
                n_class0 = np.sum(self.y == 0)
                n_class1 = np.sum(self.y == 1)
                total = len(self.y)
                print(f"  [{mode}] Class 0: {n_class0} ({n_class0 / total * 100:.1f}%)")
                print(f"  [{mode}] Class 1: {n_class1} ({n_class1 / total * 100:.1f}%)")

                unique_labels = new_unique
                n_classes = 2

            # ========== 处理标签范围不是[0,1]的情况 ==========
            if n_classes == 2:
                # 检查是否需要重新映射
                if not np.array_equal(unique_labels, np.array([0, 1])):
                    print(f"  [{mode}] Remapping {unique_labels} to [0, 1]")
                    # 建立映射
                    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
                    self.y = np.array([label_map[val] for val in self.y])
                    unique_labels = np.unique(self.y)
                    print(f"  [{mode}] After remapping: {unique_labels}")

            # ========== 最终验证 ==========
            if n_classes != 2:
                raise ValueError(
                    f"[{mode}] Binary classification requires 2 classes, "
                    f"but still have {n_classes} after processing!"
                )

            final_unique = np.unique(self.y)
            if not np.array_equal(final_unique, np.array([0, 1])):
                raise ValueError(
                    f"[{mode}] Expected labels [0, 1] but got {final_unique}"
                )

            # 显示最终状态
            n_class0 = np.sum(self.y == 0)
            n_class1 = np.sum(self.y == 1)
            print(f"  [{mode}] ✓ Final labels: {final_unique}")
            print(f"  [{mode}] ✓ Distribution: 0={n_class0}, 1={n_class1}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sig = self.x[idx]
        label = self.y[idx]

        sig_tensor = torch.from_numpy(sig).float()

        if self.mode == 'train' and self.noise_std > 0:
            noise = torch.randn_like(sig_tensor) * self.noise_std
            sig_tensor = sig_tensor + noise

        if self.is_classification:
            label_tensor = torch.tensor(label).long()
        else:
            label_tensor = torch.tensor(label).float()

        return sig_tensor, label_tensor


def load_ppgbp_kfold_data(data_dir, fold_idx, task_name='sbp', sampling_rate=50):
    """
    加载K折交叉验证的数据

    Args:
        data_dir: 数据目录
        fold_idx: 当前fold索引 (0-4)
        task_name: 任务名称
        sampling_rate: 采样率

    Returns:
        train_x, train_y, test_x, test_y
    """
    valid_tasks = ['sbp', 'dbp', 'hr', 'hyper']
    task_name = task_name.lower()
    if task_name not in valid_tasks:
        raise ValueError(f"Invalid task_name '{task_name}'. Must be one of {valid_tasks}")

    label_suffix_map = {
        'sbp': 'y_sysbp',
        'dbp': 'y_diasbp',
        'hr': 'y_hr',
        'hyper': 'y_ht'
    }
    label_suffix = label_suffix_map[task_name]

    print(f"\nLoading PPGBP Data - Fold {fold_idx}")
    print(f"Task: {task_name.upper()}")
    print(f"Data directory: {data_dir}")
    print(f"Sampling rate: {sampling_rate} Hz")

    # 构建文件路径
    fold_dir = os.path.join(data_dir, 'folds')
    train_x_path = os.path.join(fold_dir, f"fold{fold_idx}_train_X_ppg_{sampling_rate}Hz.npy")
    train_y_path = os.path.join(fold_dir, f"fold{fold_idx}_train_{label_suffix}.npy")
    test_x_path = os.path.join(fold_dir, f"fold{fold_idx}_test_X_ppg_{sampling_rate}Hz.npy")
    test_y_path = os.path.join(fold_dir, f"fold{fold_idx}_test_{label_suffix}.npy")

    # 检查文件是否存在
    required_files = [train_x_path, train_y_path, test_x_path, test_y_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(
            f"Missing files: {[os.path.basename(f) for f in missing_files]}\n"
            f"Please check fold directory: {fold_dir}"
        )

    # 加载数据
    train_x = np.load(train_x_path)
    train_y = np.load(train_y_path)
    test_x = np.load(test_x_path)
    test_y = np.load(test_y_path)

    print(f"Data loaded successfully:")
    print(f"  Train: X={train_x.shape}, y={train_y.shape}")
    print(f"  Test:  X={test_x.shape}, y={test_y.shape}")

    return train_x, train_y, test_x, test_y


def calculate_regression_metrics(preds, targets):
    """计算回归任务指标: MAE, RMSE, Correlation"""
    # 过滤掉可能的nan或inf值
    valid_mask = np.isfinite(preds) & np.isfinite(targets)
    preds_valid = preds[valid_mask]
    targets_valid = targets[valid_mask]

    if len(preds_valid) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'corr': np.nan}

    mae = np.mean(np.abs(preds_valid - targets_valid))
    rmse = np.sqrt(np.mean((preds_valid - targets_valid) ** 2))

    # 计算Pearson相关系数，添加更多保护措施
    corr = np.nan
    if len(preds_valid) > 1:
        try:
            # 检查标准差是否为0
            preds_std = np.std(preds_valid)
            targets_std = np.std(targets_valid)

            if preds_std > 1e-10 and targets_std > 1e-10:
                corr_value, p_value = pearsonr(preds_valid, targets_valid)
                # 再次检查结果是否有效
                if np.isfinite(corr_value):
                    corr = corr_value
                else:
                    print(f"  Warning: Correlation is {corr_value}, setting to nan")
            else:
                print(
                    f"  Warning: Standard deviation too small (preds_std={preds_std:.2e}, targets_std={targets_std:.2e})")
        except Exception as e:
            print(f"  Warning: Could not calculate correlation: {e}")

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'corr': corr
    }
    return metrics


def calculate_classification_metrics(preds, targets, pred_probs=None):
    """
    计算分类任务指标: Accuracy, F1, AUROC

    Args:
        preds: 预测的类别标签
        targets: 真实的类别标签
        pred_probs: 预测的概率（用于计算AUROC）
    """
    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted', zero_division=0)

    # 计算AUROC
    auroc = 0.0
    if pred_probs is not None:
        try:
            # 对于二分类问题
            if pred_probs.shape[1] == 2:
                auroc = roc_auc_score(targets, pred_probs[:, 1])
            # 对于多分类问题
            else:
                auroc = roc_auc_score(targets, pred_probs, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"Warning: Could not calculate AUROC: {e}")
            auroc = 0.0

    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'auroc': auroc
    }
    return metrics


def get_task_info(task_name):
    """获取任务信息"""
    task_info = {
        'sbp': {'unit': 'mmHg', 'type': 'regression', 'full_name': 'Systolic BP'},
        'dbp': {'unit': 'mmHg', 'type': 'regression', 'full_name': 'Diastolic BP'},
        'hr': {'unit': 'bpm', 'type': 'regression', 'full_name': 'Heart Rate'},
        'hyper': {'unit': 'class', 'type': 'classification', 'full_name': 'Hypertension'}
    }
    return task_info.get(task_name.lower(), {
        'unit': '', 'type': 'unknown', 'full_name': task_name
    })