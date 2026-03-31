"""
人体识别（Human Identification）工具类
基于PPG信号的35类分类任务

主要功能：
1. 数据增强策略（适配分类任务）
2. 多类分类评估指标（AUC, F1-Score, Accuracy）
3. 数据加载和处理
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import label_binarize


class HumanIDDataset(Dataset):
    """
    人体识别数据集类，支持数据增强
    
    Args:
        x_data: PPG信号数据，shape (N, 1, 300)
        y_data: 受试者ID标签，shape (N,), 值为0-34
        mode: 'train' 或 'test'
        aug_config: 数据增强配置字典
    """
    
    def __init__(self, x_data, y_data, mode='train', aug_config=None):
        self.x = x_data
        self.y = y_data
        self.mode = mode
        
        # 默认增强配置（针对分类任务调整）
        self.aug_config = {
            'time_warp_prob': 0.3,
            'time_warp_range': (0.95, 1.05),
            'amplitude_scale_prob': 0.5,
            'amplitude_scale_range': (0.9, 1.1),
            'baseline_wander_prob': 0.3,
            'baseline_wander_freq': (0.05, 0.15),
            'gaussian_noise_prob': 0.4,
            'gaussian_noise_std': 0.05,
            'random_mask_prob': 0.2,
            'random_mask_ratio': 0.05,
            'frequency_noise_prob': 0.2,
        }
        
        # 更新配置
        if aug_config is not None:
            self.aug_config.update(aug_config)
    
    def __len__(self):
        return len(self.x)
    
    def apply_time_warp(self, sig_tensor):
        """时间扭曲：模拟不同的生理节律变化"""
        warp_range = self.aug_config['time_warp_range']
        stretch_factor = warp_range[0] + torch.rand(1) * (warp_range[1] - warp_range[0])
        
        orig_len = sig_tensor.shape[-1]
        new_len = int(orig_len * stretch_factor)
        
        # 插值到新长度
        sig_warped = torch.nn.functional.interpolate(
            sig_tensor.unsqueeze(0),
            size=new_len,
            mode='linear',
            align_corners=False
        ).squeeze(0)
        
        # 裁剪或填充回原始长度
        if new_len > orig_len:
            sig_warped = sig_warped[..., :orig_len]
        else:
            pad_len = orig_len - new_len
            sig_warped = torch.nn.functional.pad(sig_warped, (0, pad_len), mode='reflect')
        
        return sig_warped
    
    def apply_amplitude_scale(self, sig_tensor):
        """幅度缩放：模拟不同的信号强度"""
        scale_range = self.aug_config['amplitude_scale_range']
        scale = scale_range[0] + torch.rand(1) * (scale_range[1] - scale_range[0])
        return sig_tensor * scale
    
    def apply_baseline_wander(self, sig_tensor):
        """基线漂移：添加低频正弦波"""
        freq_range = self.aug_config['baseline_wander_freq']
        baseline_freq = freq_range[0] + torch.rand(1) * (freq_range[1] - freq_range[0])
        
        length = sig_tensor.shape[-1]
        t = torch.linspace(0, 2 * np.pi, length)
        
        phase = torch.rand(1) * 2 * np.pi
        amplitude = 0.05 + torch.rand(1) * 0.1
        
        baseline = amplitude * torch.sin(baseline_freq * t + phase)
        return sig_tensor + baseline
    
    def apply_gaussian_noise(self, sig_tensor):
        """高斯噪声：模拟传感器噪声"""
        noise_std = self.aug_config['gaussian_noise_std']
        noise = torch.randn_like(sig_tensor) * noise_std
        return sig_tensor + noise
    
    def apply_random_mask(self, sig_tensor):
        """随机遮蔽：模拟信号丢失"""
        mask_ratio = self.aug_config['random_mask_ratio']
        length = sig_tensor.shape[-1]
        mask_len = int(length * mask_ratio)
        
        if mask_len > 0:
            mask_start = torch.randint(0, length - mask_len, (1,))
            sig_tensor = sig_tensor.clone()
            sig_tensor[..., mask_start:mask_start + mask_len] = 0
        
        return sig_tensor
    
    def apply_frequency_noise(self, sig_tensor):
        """频域噪声：添加高频成分"""
        length = sig_tensor.shape[-1]
        high_freq = 10 + torch.rand(1) * 20
        t = torch.linspace(0, 2 * np.pi, length)
        
        phase = torch.rand(1) * 2 * np.pi
        amplitude = 0.02 + torch.rand(1) * 0.03
        
        high_freq_noise = amplitude * torch.sin(high_freq * t + phase)
        return sig_tensor + high_freq_noise
    
    def __getitem__(self, idx):
        sig = self.x[idx]
        label = self.y[idx]
        
        sig_tensor = torch.from_numpy(sig).float()
        label_tensor = torch.tensor(label).long()  # 分类任务使用long类型
        
        if self.mode == 'train':
            # 数据增强策略
            if torch.rand(1) < self.aug_config['time_warp_prob']:
                sig_tensor = self.apply_time_warp(sig_tensor)
            
            if torch.rand(1) < self.aug_config['amplitude_scale_prob']:
                sig_tensor = self.apply_amplitude_scale(sig_tensor)
            
            if torch.rand(1) < self.aug_config['baseline_wander_prob']:
                sig_tensor = self.apply_baseline_wander(sig_tensor)
            
            if torch.rand(1) < self.aug_config['gaussian_noise_prob']:
                sig_tensor = self.apply_gaussian_noise(sig_tensor)
            
            if torch.rand(1) < self.aug_config['frequency_noise_prob']:
                sig_tensor = self.apply_frequency_noise(sig_tensor)
            
            if torch.rand(1) < self.aug_config['random_mask_prob']:
                sig_tensor = self.apply_random_mask(sig_tensor)
        
        return sig_tensor, label_tensor


def load_humanid_data(data_dir):
    """
    加载人体识别数据
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        X_all: 所有PPG信号，shape (N, 1, 300)
        y_all: 所有标签，shape (N,)，值为0-34
        subject_info: 每个样本对应的受试者信息
    """
    all_files = glob.glob(os.path.join(data_dir, "S*_x.npy"))
    if len(all_files) == 0:
        raise ValueError(f"No .npy files found in {data_dir}")
    
    X_list = []
    y_list = []
    subject_info = []
    
    print(f"Loading Human Identification Data...")
    
    # 提取所有受试者ID
    subjects = sorted(list(set([
        os.path.basename(f).split('_')[0] for f in all_files
    ])))
    
    for subj_idx, subj in enumerate(subjects):
        x_path = os.path.join(data_dir, f"{subj}_x.npy")
        y_path = os.path.join(data_dir, f"{subj}_y.npy")
        
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            continue
        
        x = np.load(x_path)
        y = np.load(y_path)
        
        # 确保标签是受试者索引
        if len(y.shape) == 0 or (len(y.shape) == 1 and len(y) == 1):
            # 如果是单个标签值，扩展到所有样本
            y = np.full(len(x), subj_idx, dtype=np.int64)
        
        X_list.append(x)
        y_list.append(y)
        subject_info.extend([subj] * len(x))
    
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    
    print(f"Data Loading Complete:")
    print(f"  Total Subjects: {len(subjects)}")
    print(f"  Total Samples: {len(X_all)}")
    print(f"  Signal Shape: {X_all.shape}")
    print(f"  Label Range: {y_all.min()} - {y_all.max()}")
    
    return X_all, y_all, subject_info


def calculate_classification_metrics(y_true, y_pred, y_prob=None, num_classes=35):
    """
    计算分类任务的评估指标
    
    Args:
        y_true: 真实标签，shape (N,)
        y_pred: 预测标签，shape (N,)
        y_prob: 预测概率，shape (N, num_classes)，用于计算AUC
        num_classes: 类别数量
    
    Returns:
        metrics: 包含各种指标的字典
    """
    # 1. Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # 2. F1-Score (macro average)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # 3. AUC (需要概率输出)
    if y_prob is not None:
        try:
            # One-vs-Rest AUC
            y_true_bin = label_binarize(y_true, classes=range(num_classes))
            
            # 如果只有部分类别出现在测试集中
            if y_true_bin.shape[1] < num_classes:
                # 补齐缺失的类别列
                full_y_true_bin = np.zeros((len(y_true), num_classes))
                unique_classes = np.unique(y_true)
                for i, cls in enumerate(unique_classes):
                    full_y_true_bin[:, cls] = y_true_bin[:, i]
                y_true_bin = full_y_true_bin
            
            # Macro AUC
            auc_macro = roc_auc_score(
                y_true_bin, y_prob, 
                average='macro', 
                multi_class='ovr'
            )
            # Weighted AUC
            auc_weighted = roc_auc_score(
                y_true_bin, y_prob, 
                average='weighted', 
                multi_class='ovr'
            )
        except Exception as e:
            print(f"Warning: AUC calculation failed: {e}")
            auc_macro = 0.0
            auc_weighted = 0.0
    else:
        auc_macro = 0.0
        auc_weighted = 0.0
    
    # 4. Per-class F1 scores
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
        'per_class_f1': per_class_f1
    }
    
    return metrics


def print_classification_report(y_true, y_pred, target_names=None):
    """打印详细的分类报告"""
    print("\nClassification Report:")
    print("=" * 60)
    if target_names is None:
        target_names = [f"Subject_{i}" for i in range(35)]
    
    print(classification_report(
        y_true, y_pred, 
        target_names=target_names,
        zero_division=0,
        digits=4
    ))
    
    # 混淆矩阵统计
    cm = confusion_matrix(y_true, y_pred)
    correct_per_class = np.diag(cm)
    total_per_class = cm.sum(axis=1)
    
    print("\nPer-Class Accuracy:")
    print("-" * 60)
    for i in range(len(total_per_class)):
        if total_per_class[i] > 0:
            acc = correct_per_class[i] / total_per_class[i] * 100
            print(f"  {target_names[i]}: {acc:.2f}% ({correct_per_class[i]}/{total_per_class[i]})")


class AugmentationConfigs:
    """预定义的增强配置"""
    
    # 轻度增强（推荐用于人体识别）
    LIGHT = {
        'time_warp_prob': 0.2,
        'time_warp_range': (0.97, 1.03),
        'amplitude_scale_prob': 0.3,
        'amplitude_scale_range': (0.95, 1.05),
        'baseline_wander_prob': 0.2,
        'gaussian_noise_prob': 0.3,
        'gaussian_noise_std': 0.03,
        'random_mask_prob': 0.1,
        'frequency_noise_prob': 0.1,
    }
    
    # 中度增强
    MEDIUM = {
        'time_warp_prob': 0.3,
        'time_warp_range': (0.95, 1.05),
        'amplitude_scale_prob': 0.5,
        'amplitude_scale_range': (0.9, 1.1),
        'baseline_wander_prob': 0.3,
        'gaussian_noise_prob': 0.4,
        'gaussian_noise_std': 0.05,
        'random_mask_prob': 0.2,
        'frequency_noise_prob': 0.2,
    }
    
    # 强度增强
    STRONG = {
        'time_warp_prob': 0.4,
        'time_warp_range': (0.90, 1.10),
        'amplitude_scale_prob': 0.6,
        'amplitude_scale_range': (0.85, 1.15),
        'baseline_wander_prob': 0.4,
        'gaussian_noise_prob': 0.5,
        'gaussian_noise_std': 0.08,
        'random_mask_prob': 0.3,
        'frequency_noise_prob': 0.3,
    }
