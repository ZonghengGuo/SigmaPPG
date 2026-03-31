import logging
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="w")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(filepath)
    return logger


class Dataset_train(Dataset):
    def __init__(self, signal_train, y_train, mode='train'):
        self.strain = signal_train
        self.ytrain = y_train
        self.mode = mode
        self.sig_len = 1250  # 25s * 50Hz

        # 打印数据形状以便调试
        print(f"Dataset initialized with signal shape: {self.strain.shape}, label shape: {self.ytrain.shape}")

    def __len__(self):
        return len(self.ytrain)

    def __getitem__(self, index):
        sig = self.strain[index]

        # 确保内存是可写的且连续的
        if isinstance(sig, np.ndarray):
            sig = torch.from_numpy(sig.copy()).float()
        elif not isinstance(sig, torch.Tensor):
            sig = torch.tensor(sig).float()

        # 🔧 关键修复：处理多维数据，确保变成 1D
        # 如果数据是多维的（例如 [25, 52] 或 [1, 25, 52]），先 flatten
        if sig.ndim > 1:
            sig = sig.flatten()

        # 🔧 关键修复：确保长度正确
        current_len = len(sig)
        if current_len != self.sig_len:
            print(f"Warning: Signal length mismatch! Expected {self.sig_len}, got {current_len}. Fixing...")
            if current_len > self.sig_len:
                # 截断
                sig = sig[:self.sig_len]
            else:
                # 填充零
                padding = torch.zeros(self.sig_len - current_len)
                sig = torch.cat([sig, padding])

        # 防止 NaN 输入导致 Loss NaN
        if torch.isnan(sig).any() or torch.isinf(sig).any():
            sig = torch.nan_to_num(sig, nan=0.0, posinf=1.0, neginf=-1.0)

        # 训练时数据增强
        if self.mode == 'train':
            # 幅值缩放
            if random.random() < 0.5:
                scale_factor = random.uniform(0.9, 1.1)
                sig = sig * scale_factor

            # 添加高斯噪声
            if random.random() < 0.5:
                noise = torch.randn_like(sig) * 0.05
                sig = sig + noise
        sig = sig.unsqueeze(0)

        label = self.ytrain[index]
        if isinstance(label, np.ndarray) or isinstance(label, np.integer):
            label = torch.tensor(label).long()

        return sig, label