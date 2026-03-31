"""
改进版BIDMC工具类 - 增强的数据增强策略
可以直接替换原有的tools.py的BIDMCDataset类

主要改进：
1. 时间扭曲 (Time Warping)
2. 基线漂移 (Baseline Wander)
3. 随机遮蔽 (Random Masking)
4. 更丰富的噪声模式
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob


class BIDMCDataset(Dataset):
    """
    增强版BIDMC数据集类，支持更丰富的数据增强

    Args:
        x_data: PPG信号数据
        y_data: 标签数据（可以是单个标签或标签字典）
        task_name: 任务名称 ('rr', 'hr', 'spo2')
        mode: 'train' 或 'test'
        aug_config: 数据增强配置字典
    """

    def __init__(self, x_data, y_data, task_name='rr', mode='train', aug_config=None):
        self.x = x_data
        self.task_name = task_name
        self.mode = mode

        # 默认增强配置
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

        # 支持字典形式的多标签
        if isinstance(y_data, dict):
            self.y = y_data.get(task_name, None)
            if self.y is None:
                raise ValueError(f"Task '{task_name}' not found in labels")
        else:
            self.y = y_data

    def __len__(self):
        return len(self.x)

    def apply_time_warp(self, sig_tensor):
        """
        时间扭曲：模拟不同的生理节律变化
        拉伸或压缩时间轴，然后插值回原始长度
        """
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
        """
        基线漂移：添加低频正弦波模拟呼吸引起的基线变化
        """
        freq_range = self.aug_config['baseline_wander_freq']
        baseline_freq = freq_range[0] + torch.rand(1) * (freq_range[1] - freq_range[0])

        length = sig_tensor.shape[-1]
        t = torch.linspace(0, 2 * np.pi, length)

        # 随机相位和幅度
        phase = torch.rand(1) * 2 * np.pi
        amplitude = 0.05 + torch.rand(1) * 0.1  # 0.05-0.15

        baseline = amplitude * torch.sin(baseline_freq * t + phase)
        return sig_tensor + baseline

    def apply_gaussian_noise(self, sig_tensor):
        """高斯噪声：模拟传感器噪声"""
        noise_std = self.aug_config['gaussian_noise_std']
        noise = torch.randn_like(sig_tensor) * noise_std
        return sig_tensor + noise

    def apply_random_mask(self, sig_tensor):
        """
        随机遮蔽：模拟信号丢失或干扰
        将信号的某个随机片段置零
        """
        mask_ratio = self.aug_config['random_mask_ratio']
        length = sig_tensor.shape[-1]
        mask_len = int(length * mask_ratio)

        if mask_len > 0:
            mask_start = torch.randint(0, length - mask_len, (1,))
            sig_tensor = sig_tensor.clone()
            sig_tensor[..., mask_start:mask_start + mask_len] = 0

        return sig_tensor

    def apply_frequency_noise(self, sig_tensor):
        """
        频域噪声：在频域添加随机成分
        模拟电磁干扰等高频噪声
        """
        # 添加高频正弦波
        length = sig_tensor.shape[-1]
        high_freq = 10 + torch.rand(1) * 20  # 10-30 Hz的高频
        t = torch.linspace(0, 2 * np.pi, length)

        phase = torch.rand(1) * 2 * np.pi
        amplitude = 0.02 + torch.rand(1) * 0.03  # 0.02-0.05

        high_freq_noise = amplitude * torch.sin(high_freq * t + phase)
        return sig_tensor + high_freq_noise

    def __getitem__(self, idx):
        sig = self.x[idx]
        label = self.y[idx]

        sig_tensor = torch.from_numpy(sig).float()
        label_tensor = torch.tensor(label).float()

        if self.mode == 'train':
            # 🎯 增强的数据增强策略

            # 1. 时间扭曲 (Time Warping)
            if torch.rand(1) < self.aug_config['time_warp_prob']:
                sig_tensor = self.apply_time_warp(sig_tensor)

            # 2. 幅度缩放
            if torch.rand(1) < self.aug_config['amplitude_scale_prob']:
                sig_tensor = self.apply_amplitude_scale(sig_tensor)

            # 3. 基线漂移
            if torch.rand(1) < self.aug_config['baseline_wander_prob']:
                sig_tensor = self.apply_baseline_wander(sig_tensor)

            # 4. 高斯噪声
            if torch.rand(1) < self.aug_config['gaussian_noise_prob']:
                sig_tensor = self.apply_gaussian_noise(sig_tensor)

            # 5. 频域噪声
            if torch.rand(1) < self.aug_config['frequency_noise_prob']:
                sig_tensor = self.apply_frequency_noise(sig_tensor)

            # 6. 随机遮蔽 (最后应用，避免被其他增强覆盖)
            if torch.rand(1) < self.aug_config['random_mask_prob']:
                sig_tensor = self.apply_random_mask(sig_tensor)

        return sig_tensor, label_tensor


# 保留原有的功能函数（不改变）
def load_bidmc_data(data_dir, task_name='rr'):
    """
    加载 BIDMC 数据并按病人划分 Train/Test
    [原有代码保持不变]
    """
    # 检查任务名称
    valid_tasks = ['rr', 'hr', 'spo2']
    if task_name not in valid_tasks:
        raise ValueError(f"Invalid task_name '{task_name}'. Must be one of {valid_tasks}")

    # 标签文件后缀映射
    label_suffix_map = {
        'rr': '_y_rr.npy',
        'hr': '_y_hr.npy',
        'spo2': '_y_spo2.npy'
    }

    # 兼容旧格式
    all_files = glob.glob(os.path.join(data_dir, "*_x.npy"))
    if len(all_files) == 0:
        raise ValueError(f"No .npy files found in {data_dir}")

    train_x_list, train_y_list = [], []
    test_x_list, test_y_list = [], []

    print(f"Loading BIDMC Data for task: {task_name.upper()}")

    subjects = sorted(list(set([os.path.basename(f).split('_')[0] for f in all_files])))

    for subj in subjects:
        x_path = os.path.join(data_dir, f"{subj}_x.npy")
        label_suffix = label_suffix_map[task_name]
        y_path = os.path.join(data_dir, f"{subj}{label_suffix}")

        if not os.path.exists(y_path) and task_name == 'rr':
            y_path_old = os.path.join(data_dir, f"{subj}_y.npy")
            if os.path.exists(y_path_old):
                y_path = y_path_old

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            continue

        x = np.load(x_path)
        y = np.load(y_path)

        try:
            subj_id = int(subj[1:])
        except:
            subj_id = 999

        if subj_id <= 40:
            train_x_list.append(x)
            train_y_list.append(y)
        else:
            test_x_list.append(x)
            test_y_list.append(y)

    train_x = np.concatenate(train_x_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)

    if test_x_list:
        test_x = np.concatenate(test_x_list, axis=0)
        test_y = np.concatenate(test_y_list, axis=0)
    else:
        test_x, test_y = np.empty((0, 1, 4000)), np.empty((0,))

    print(f"Data Split Result:")
    print(f"  Train Samples: {len(train_x)}")
    print(f"  Test Samples:  {len(test_x)}")

    return train_x, train_y, test_x, test_y


def load_bidmc_multitask_data(data_dir, tasks=['rr', 'hr', 'spo2']):
    """
    加载多任务数据
    [原有代码保持不变]
    """
    all_files = glob.glob(os.path.join(data_dir, "*_x.npy"))
    if len(all_files) == 0:
        raise ValueError(f"No .npy files found in {data_dir}")

    train_x_list = []
    train_y_dict = {task: [] for task in tasks}
    test_x_list = []
    test_y_dict = {task: [] for task in tasks}

    subjects = sorted(list(set([os.path.basename(f).split('_')[0] for f in all_files])))

    label_suffix_map = {
        'rr': '_y_rr.npy',
        'hr': '_y_hr.npy',
        'spo2': '_y_spo2.npy'
    }

    for subj in subjects:
        x_path = os.path.join(data_dir, f"{subj}_x.npy")
        y_paths = {}
        all_exist = True

        for task in tasks:
            y_path = os.path.join(data_dir, f"{subj}{label_suffix_map[task]}")
            if os.path.exists(y_path):
                y_paths[task] = y_path
            else:
                all_exist = False
                break

        if not os.path.exists(x_path) or not all_exist:
            continue

        x = np.load(x_path)
        y_dict = {task: np.load(y_paths[task]) for task in tasks}

        try:
            subj_id = int(subj[1:])
        except:
            subj_id = 999

        if subj_id <= 40:
            train_x_list.append(x)
            for task in tasks:
                train_y_dict[task].append(y_dict[task])
        else:
            test_x_list.append(x)
            for task in tasks:
                test_y_dict[task].append(y_dict[task])

    train_x = np.concatenate(train_x_list, axis=0)
    test_x = np.concatenate(test_x_list, axis=0) if test_x_list else np.empty((0, 1, 4000))

    for task in tasks:
        train_y_dict[task] = np.concatenate(train_y_dict[task], axis=0)
        test_y_dict[task] = np.concatenate(test_y_dict[task], axis=0) if test_y_dict[task] else np.empty((0,))

    return train_x, train_y_dict, test_x, test_y_dict


def calculate_mae(preds, targets):
    """计算平均绝对误差"""
    return np.mean(np.abs(preds - targets))


def calculate_rmse(preds, targets):
    """计算均方根误差"""
    return np.sqrt(np.mean((preds - targets) ** 2))


def calculate_metrics(preds, targets, task_name='rr'):
    """计算多种评估指标"""
    mae = calculate_mae(preds, targets)
    rmse = calculate_rmse(preds, targets)
    mape = np.mean(np.abs((preds - targets) / (targets + 1e-8))) * 100
    corr = np.corrcoef(preds, targets)[0, 1]

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'corr': corr
    }

    return metrics


def get_task_unit(task_name):
    """获取任务的单位"""
    units = {
        'rr': 'breaths/min',
        'hr': 'beats/min',
        'spo2': '%'
    }
    return units.get(task_name, '')


# 向后兼容：提供原始Dataset类的别名
BIDMCDataset = BIDMCDataset


# 使用示例和配置
class AugmentationConfigs:
    """预定义的增强配置"""

    # 轻度增强（保守）
    LIGHT = {
        'time_warp_prob': 0.2,
        'time_warp_range': (0.97, 1.03),
        'amplitude_scale_prob': 0.4,
        'amplitude_scale_range': (0.95, 1.05),
        'baseline_wander_prob': 0.2,
        'gaussian_noise_prob': 0.3,
        'gaussian_noise_std': 0.03,
        'random_mask_prob': 0.1,
        'frequency_noise_prob': 0.1,
    }

    # 中度增强（推荐）
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

    # 强度增强（激进）
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