import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample, butter, sosfiltfilt
from scipy.interpolate import interp1d
import torch


class PreprocessDalia:

    def __init__(self, args):
        self.args = args
        self.dataset_path = args.raw_data_path
        self.save_path = os.path.join(args.seg_save_path, "dalia_activity_loso")

        # Original sampling rates
        self.ORIG_PPG_FS = 64
        self.ORIG_ACC_FS = 32
        self.ORIG_ACTIVITY_FS = 4  # Activity labels are at 4Hz in the pkl file

        # Target sampling rate: 50Hz
        self.TARGET_FS = 50

        # Windowing parameters (matching original DALIA paper)
        self.WINDOW_SEC = 8
        self.SHIFT_SEC = 2

        self.TARGET_LEN = int(self.WINDOW_SEC * self.TARGET_FS)  # 8s * 50Hz = 400
        self.SHIFT_LEN = int(self.SHIFT_SEC * self.TARGET_FS)    # 2s * 50Hz = 100

        self.subject_ids = [f'S{i}' for i in range(1, 16)]  # S1 to S15

        print(f"DALIA Activity Classification Preprocessor Config (LoSo):")
        print(f"  > Task: 9-class Activity Classification")
        print(f"  > Target Fs: {self.TARGET_FS}Hz")
        print(f"  > Window: {self.WINDOW_SEC}s ({self.TARGET_LEN} points)")
        print(f"  > Shift: {self.SHIFT_SEC}s ({self.SHIFT_LEN} points)")
        print(f"  > Subjects: {len(self.subject_ids)}")

    def load_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
        y = sosfiltfilt(sos, data)
        return y

    def interpolate_nans(self, data):
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                nan_mask = np.isnan(data)
                if nan_mask.all():
                    return np.zeros_like(data)
                if nan_mask.any():
                    not_nan = ~nan_mask
                    indices = np.arange(len(data))
                    data[nan_mask] = np.interp(indices[nan_mask], indices[not_nan], data[not_nan])
            elif data.ndim == 2:
                for col in range(data.shape[1]):
                    nan_mask = np.isnan(data[:, col])
                    if nan_mask.all():
                        data[:, col] = 0
                    elif nan_mask.any():
                        not_nan = ~nan_mask
                        indices = np.arange(len(data))
                        data[nan_mask, col] = np.interp(
                            indices[nan_mask], indices[not_nan], data[not_nan, col]
                        )
        return data

    def process_subject(self, subject_id):
        """Process a single subject, return (segments, labels)"""
        file_path = os.path.join(self.dataset_path, subject_id, f"{subject_id}.pkl")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found.")
            return None, None

        data = self.load_pickle(file_path)

        ppg = data['signal']['wrist']['BVP']  # 64Hz
        activity = data['activity']            # 4Hz

        print(f"\n{subject_id} - PPG: {len(ppg)} samples | "
              f"Activity: {len(activity)} samples")

        # Resample PPG to target frequency
        ppg_duration = len(ppg) / self.ORIG_PPG_FS
        target_total_len = int(ppg_duration * self.TARGET_FS)
        ppg_resampled = resample(ppg, target_total_len).flatten()

        # Bandpass filter (0.5–8 Hz)
        ppg_filtered = self.butter_bandpass_filter(
            ppg_resampled, 0.5, 8.0, self.TARGET_FS
        )
        ppg_filtered = self.interpolate_nans(ppg_filtered)

        segments = []
        segment_labels = []
        window_idx = 0

        for start_sample in range(0, len(ppg_filtered), self.SHIFT_LEN):
            end_sample = start_sample + self.TARGET_LEN

            if end_sample > len(ppg_filtered):
                break

            segment = ppg_filtered[start_sample:end_sample]

            act_start_idx = int(window_idx * self.SHIFT_SEC * self.ORIG_ACTIVITY_FS)
            act_end_idx = int(act_start_idx + self.WINDOW_SEC * self.ORIG_ACTIVITY_FS)

            if act_end_idx > len(activity):
                break

            activity_window = activity[act_start_idx:act_end_idx]

            from scipy import stats
            mode_result = stats.mode(activity_window, keepdims=False)
            label = int(mode_result[0])

            seg_min = np.min(segment)
            seg_max = np.max(segment)
            if seg_max - seg_min < 1e-9:
                segment_norm = np.zeros_like(segment)
            else:
                segment_norm = (segment - seg_min) / (seg_max - seg_min)

            segment_norm = self.interpolate_nans(segment_norm)

            segments.append(segment_norm)
            segment_labels.append(label)
            window_idx += 1

        print(f"  Generated {len(segments)} segments")
        return np.array(segments), np.array(segment_labels)

    def preprocess_save(self):
        """LoSo: 每次留出一个被试作为测试集，其余14个作为训练集"""
        os.makedirs(self.save_path, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"🚀 DaLiA LoSo 预处理")
        print(f"{'=' * 60}\n")

        # Step 1: 预处理所有被试
        print("Step 1: 预处理所有被试数据...")
        all_subject_data = {}
        total_activity_counts = np.zeros(9, dtype=int)

        for subj in tqdm(self.subject_ids, desc="Processing Subjects"):
            X, y = self.process_subject(subj)
            if X is not None and len(X) > 0:
                all_subject_data[subj] = {'X': X, 'y': y}
                activity_counts = np.bincount(y, minlength=9)
                total_activity_counts += activity_counts
                print(f"  {subj}: {len(X)} segments | dist={activity_counts.tolist()}")
            else:
                print(f"  ⚠️  {subj}: 无有效数据，跳过")

        # Step 2: LoSo 循环
        print(f"\nStep 2: 生成 LoSo 折叠（共 {len(self.subject_ids)} 折）...")

        valid_subjects = list(all_subject_data.keys())

        for fold_idx, test_subj in enumerate(valid_subjects):
            train_subjects = [s for s in valid_subjects if s != test_subj]

            # 训练集
            train_X = np.concatenate([all_subject_data[s]['X'] for s in train_subjects])
            train_y = np.concatenate([all_subject_data[s]['y'] for s in train_subjects])

            # 测试集
            test_X = all_subject_data[test_subj]['X']
            test_y = all_subject_data[test_subj]['y']

            # 添加 channel 维度 (N, 400) -> (N, 1, 400)
            train_X = train_X[:, np.newaxis, :]
            test_X = test_X[:, np.newaxis, :]

            # 保存
            fold_dir = os.path.join(self.save_path, f"fold_{fold_idx:02d}_test_{test_subj}")
            os.makedirs(fold_dir, exist_ok=True)

            np.save(os.path.join(fold_dir, "train_x.npy"), train_X.astype(np.float32))
            np.save(os.path.join(fold_dir, "train_y.npy"), train_y.astype(np.int64))
            np.save(os.path.join(fold_dir, "test_x.npy"),  test_X.astype(np.float32))
            np.save(os.path.join(fold_dir, "test_y.npy"),  test_y.astype(np.int64))

            print(f"  Fold {fold_idx:02d} | test={test_subj} | "
                  f"train={len(train_X):5d} | test={len(test_X):4d} | "
                  f"test dist={np.bincount(test_y, minlength=9).tolist()}")

        print(f"\n{'=' * 60}")
        print(f"✅ DaLiA LoSo 预处理完成！共 {len(valid_subjects)} 折")
        print(f"数据已保存到: {self.save_path}")
        activity_names = ['Transient', 'Sitting', 'Stairs', 'Table Soccer',
                          'Cycling', 'Driving', 'Lunch', 'Walking', 'Working']
        print(f"\n全体活动分布:")
        for i, (name, count) in enumerate(zip(activity_names, total_activity_counts)):
            print(f"  {i}: {name:15s} - {count:5d} ({count / total_activity_counts.sum() * 100:.1f}%)")
        print(f"{'=' * 60}\n")