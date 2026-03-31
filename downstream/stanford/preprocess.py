import os
import numpy as np
import torch
from scipy.signal import butter, resample, sosfiltfilt
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing


class PreprocessStanford:
    def __init__(self, args):
        self.args = args
        self.dataset_path = args.raw_data_path
        self.SRC_FREQ = 32
        self.TARGET_FREQ = 50  # 修改为 50Hz

        # 目标时长 25秒
        self.TARGET_DURATION = 25
        # 25 * 50 = 1250 samples
        self.TOTAL_LEN = int(self.TARGET_DURATION * self.TARGET_FREQ)

    def process_single_signal(self, sig):
        try:
            sos = butter(2, [0.5, 8], btype='band', fs=self.SRC_FREQ, output='sos')
            filtered = sosfiltfilt(sos, sig)

            if len(filtered) == 0:
                return np.zeros(self.TOTAL_LEN, dtype=np.float32)

            resampled = resample(filtered, self.TOTAL_LEN)

            sig_min = np.min(resampled)
            sig_max = np.max(resampled)
            if sig_max - sig_min < 1e-9:
                final_sig = np.zeros_like(resampled)
            else:
                final_sig = (resampled - sig_min) / (sig_max - sig_min)

            return final_sig.astype(np.float32)

        except Exception as e:
            print(f"Error processing signal: {e}")
            return np.zeros(self.TOTAL_LEN, dtype=np.float32)

    def _process_split(self, split_name, file_name):
        file_path = os.path.join(self.dataset_path, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            return None, None

        print(f"Loading raw data for {split_name}...")
        data = np.load(file_path, allow_pickle=True)

        keys = data.files
        sig_key = 'signal' if 'signal' in keys else 'x'
        qa_key = 'qa_label'

        raw_signals = data[sig_key]

        # 处理 QA 标签
        if qa_key in keys:
            raw_qa = data[qa_key]
        else:
            print(f"Warning: No qa_label found in {split_name}, using zeros")
            raw_qa = np.zeros((len(raw_signals), 3))

        if raw_signals.ndim == 3:
            raw_signals = np.squeeze(raw_signals)

        # 标签转索引
        qa_indices = np.argmax(raw_qa, axis=1) if raw_qa.ndim == 2 else raw_qa

        print(f"Processing {len(raw_signals)} signals for {split_name} using Parallel CPU...")

        # 并行处理
        processed_samples = Parallel(n_jobs=-1)(
            delayed(self.process_single_signal)(sig)
            for sig in tqdm(raw_signals, desc=f"Converting {split_name}")
        )

        samples_np = np.array(processed_samples, dtype=np.float32)
        qa_np = np.array(qa_indices, dtype=np.int64)

        return samples_np, qa_np

    def preprocess_save(self):
        files_map = {
            'train': 'train.npz',
            'val': 'validate.npz',
            'test': 'test.npz'
        }

        output_dir = os.path.join(self.dataset_path, "out/quality_50hz")
        os.makedirs(output_dir, exist_ok=True)

        for split, fname in files_map.items():
            samples, qa_labels = self._process_split(split, fname)

            if samples is not None and len(samples) > 0:
                print(f"Saving {split} set to .npy files...")
                np.save(os.path.join(output_dir, f"{split}_x.npy"), samples)
                np.save(os.path.join(output_dir, f"{split}_y_qa.npy"), qa_labels)
                print(f"Saved {split}. Shape: {samples.shape}, Labels: {qa_labels.shape}")