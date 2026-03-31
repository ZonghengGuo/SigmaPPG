import os
import pickle
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, lfilter
from scipy import fftpack
from typing import List, Tuple


class PreprocessWESAD:
    """
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9358140
    Leave-One-Subject-Out (LoSo) cross-validation protocol.
    """

    def __init__(self, wesad_args):
        self.FS_ORIG = 64
        self.FS_NEW = getattr(wesad_args, 'sampling_rate', 50)

        self.dataset_path = wesad_args.raw_data_path
        self.raw_data_path = os.path.join(self.dataset_path, "WESAD")

        self.task_name = wesad_args.task_name
        if self.task_name == 'binary':
            self.output_path = os.path.join(self.dataset_path, "WESAD/out/binary_loso")
        else:
            self.output_path = os.path.join(self.dataset_path, "WESAD/out/multiclass_loso")

        self.window_seconds = 60
        self.subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

    def butter_bandpass(self, lowcut: float, highcut: float, fs: float, order: int = 5) -> Tuple[
        np.ndarray, np.ndarray]:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpassfilter(self, data: np.ndarray, lowcut: float, highcut: float, fs: float,
                              order: int = 5) -> np.ndarray:
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def movingaverage(self, data: np.ndarray, size: int = 4) -> np.ndarray:
        data_set = np.asarray(data)
        weights = np.ones(size) / size
        return np.convolve(data_set, weights, mode='valid')

    def FFT(self, y: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray, float]:
        N = len(y)
        T = 1.0 / fs
        yf = fftpack.fft(y)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        return xf, 2.0 / N * np.abs(yf[:N // 2]), N / fs

    def threshold_peakdetection(self, signal: np.ndarray, fs: int) -> List[int]:
        signal = np.array([item.real for item in signal])
        ths = (max(signal) + np.mean(signal)) / 2
        peak = []

        for i in range(len(signal) - 1):
            if signal[i] < ths < signal[i + 1]:
                peak.append(i)

        peaktopeak = fs * 60 / 200
        cleaned_peak = [peak[0]] if peak else []

        for p in peak[1:]:
            if p - cleaned_peak[-1] > peaktopeak:
                cleaned_peak.append(p)

        return cleaned_peak

    def RR_interval(self, peak: List[int], fs: int) -> List[float]:
        RR = []
        for i in range(len(peak) - 1):
            RR.append((peak[i + 1] - peak[i]) / fs * 1000)
        return RR

    def calc_heartrate(self, RR_list: List[float]) -> List[float]:
        HR = []
        for val in RR_list:
            if 400 < val < 1500:
                HR.append(60 * 1000 / val)
        return HR if HR else [70]

    def get_cutoff(self, block: np.ndarray, fs: int) -> List[float]:
        block = np.array([item.real for item in block])

        try:
            peak = self.threshold_peakdetection(block, fs)
            hr_mean = np.mean(self.calc_heartrate(self.RR_interval(peak, fs)))
            low_cutoff = max(0.5, np.round(hr_mean / 60 - 0.6, 1))
        except:
            low_cutoff = 0.5

        frequencies, fourierTransform, timePeriod = self.FFT(block, fs)
        ths = max(abs(fourierTransform)) * 0.1

        high_cutoff = 8.0
        for i in range(int(5 * timePeriod), 0, -1):
            if i < len(fourierTransform) and abs(fourierTransform[i]) > ths:
                high_cutoff = np.round(i / timePeriod, 1)
                break

        return [low_cutoff, min(high_cutoff, 10.0)]

    def compute_and_reconstruction_dft(self, ppg: np.ndarray, fs: int, sec: int, overlap: int,
                                       cutoff: List[float]) -> np.ndarray:
        N = fs * sec
        result = []

        for i in range(0, len(ppg), N - overlap):
            if i + N > len(ppg):
                break

            segment = ppg[i:i + N]
            yf = fftpack.fft(segment)
            freq = fftpack.fftfreq(len(segment), 1 / fs)

            yf_filtered = yf.copy()
            mask = (np.abs(freq) < cutoff[0]) | (np.abs(freq) > cutoff[1])
            yf_filtered[mask] = 0

            reconstructed = np.real(fftpack.ifft(yf_filtered))

            if i == 0:
                result.extend(reconstructed)
            else:
                result.extend(reconstructed[overlap:])

        return np.array(result)

    def resample_linear(self, signal: np.ndarray, orig_hz: int, new_hz: int) -> np.ndarray:
        if orig_hz == new_hz:
            return signal

        orig_length = len(signal)
        new_length = int(orig_length * new_hz / orig_hz)

        orig_indices = np.arange(orig_length)
        new_indices = np.linspace(0, orig_length - 1, new_length)

        resampled = np.interp(new_indices, orig_indices, signal)
        return resampled

    def denoise_ppg(self, ppg: np.ndarray) -> np.ndarray:
        ppg_bp = self.butter_bandpassfilter(ppg, 0.5, 10, self.FS_ORIG, order=2)

        signal_one_percent = int(len(ppg_bp))
        cutoff = self.get_cutoff(ppg_bp[:signal_one_percent], self.FS_ORIG)

        sec = 12
        N = self.FS_ORIG * sec
        overlap = int(np.round(N * 0.02))
        ppg_freq = self.compute_and_reconstruction_dft(ppg_bp, self.FS_ORIG, sec, overlap, cutoff)

        if len(ppg_freq) > len(ppg):
            ppg_freq = ppg_freq[:len(ppg)]
        elif len(ppg_freq) < len(ppg):
            ppg_freq = np.pad(ppg_freq, (0, len(ppg) - len(ppg_freq)), mode='edge')

        try:
            fwd = self.movingaverage(ppg_freq, size=3)
            bwd = self.movingaverage(ppg_freq[::-1], size=3)
            ppg_ma = np.mean(np.vstack((fwd, bwd[::-1])), axis=0)
        except:
            ppg_ma = ppg_freq

        ppg_real = np.real(ppg_ma)

        if self.FS_NEW != self.FS_ORIG:
            ppg_resampled = self.resample_linear(ppg_real, self.FS_ORIG, self.FS_NEW)
        else:
            ppg_resampled = ppg_real

        sig_min = np.min(ppg_resampled)
        sig_max = np.max(ppg_resampled)
        if sig_max - sig_min < 1e-9:
            ppg_normalized = np.zeros_like(ppg_resampled)
        else:
            ppg_normalized = (ppg_resampled - sig_min) / (sig_max - sig_min)

        return ppg_normalized

    def get_label_mapping(self):
        if self.task_name == 'binary':
            return {1: 0, 2: 1, 3: 0, 4: 0}
        else:
            return {1: 0, 2: 1, 3: 2, 4: 3}

    def process_subject(self, subject_id: int, label_mapping: dict) -> Tuple[List, List]:
        """处理单个被试，返回该被试的所有样本和标签"""
        pkl_path = os.path.join(self.raw_data_path, f"S{subject_id}", f"S{subject_id}.pkl")

        if not os.path.exists(pkl_path):
            print(f"⚠️  文件不存在: {pkl_path}")
            return [], []

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        ppg_raw = data['signal']['wrist']['BVP'].flatten()
        labels_700 = data['label']

        label_indices = np.linspace(0, len(labels_700) - 1, len(ppg_raw)).astype(int)
        labels_sync = labels_700[label_indices]

        samples, labels = [], []
        target_labels = [1, 2, 3, 4]

        for label_val in target_labels:
            mask = (labels_sync == label_val)
            idx = np.where(mask)[0]

            if len(idx) == 0:
                continue

            splits = np.where(np.diff(idx) > 1)[0] + 1
            segments = np.split(idx, splits)

            for seg_idx in segments:
                if len(seg_idx) < self.FS_ORIG * self.window_seconds:
                    continue

                ppg_seg = ppg_raw[seg_idx]

                try:
                    ppg_denoised = self.denoise_ppg(ppg_seg)
                except Exception as e:
                    print(f"⚠️  S{subject_id} 标签 {label_val} 去噪失败: {e}")
                    continue

                window_size = self.FS_NEW * self.window_seconds

                for start in range(0, len(ppg_denoised) - window_size + 1, window_size):
                    ppg_window = ppg_denoised[start:start + window_size]
                    samples.append(ppg_window)
                    labels.append(label_mapping[label_val])

        return samples, labels

    def preprocess_save(self):
        """LoSo 预处理：每次留出一个被试作为测试集，其余作为训练集"""
        os.makedirs(self.output_path, exist_ok=True)
        label_mapping = self.get_label_mapping()

        print(f"\n{'=' * 60}")
        print(f"🚀 WESAD LoSo 预处理（{self.task_name.upper()}）")
        print(f"{'=' * 60}")
        print(f"原始采样率: {self.FS_ORIG}Hz → 目标: {self.FS_NEW}Hz")
        print(f"窗口大小: {self.window_seconds}s | 被试数: {len(self.subject_ids)}")
        print(f"{'=' * 60}\n")

        # 预先处理所有被试，避免重复 I/O
        print("Step 1: 预处理所有被试数据...")
        all_subject_data = {}
        for subject_id in tqdm(self.subject_ids, desc="处理被试"):
            samples, labels = self.process_subject(subject_id, label_mapping)
            all_subject_data[subject_id] = {
                'samples': samples,
                'labels': labels
            }
            print(f"  S{subject_id}: {len(samples)} 个窗口")

        # LoSo 循环：每次留出一个被试
        print("\nStep 2: 生成 LoSo 折叠...")
        fold_aucs = []

        for fold_idx, test_subject in enumerate(self.subject_ids):
            train_subjects = [s for s in self.subject_ids if s != test_subject]

            # 训练集：合并所有其他被试
            train_samples, train_labels = [], []
            for s in train_subjects:
                train_samples.extend(all_subject_data[s]['samples'])
                train_labels.extend(all_subject_data[s]['labels'])

            # 测试集：留出被试
            test_samples = all_subject_data[test_subject]['samples']
            test_labels = all_subject_data[test_subject]['labels']

            if len(test_samples) == 0:
                print(f"  ⚠️  Fold {fold_idx} (test=S{test_subject}): 无测试数据，跳过")
                continue

            # 保存当前折叠
            fold_dir = os.path.join(self.output_path, f"fold_{fold_idx:02d}_test_S{test_subject}")
            os.makedirs(fold_dir, exist_ok=True)

            np.save(os.path.join(fold_dir, "train_x.npy"),
                    np.array(train_samples, dtype=np.float32))
            np.save(os.path.join(fold_dir, "train_y.npy"),
                    np.array(train_labels, dtype=np.int64))
            np.save(os.path.join(fold_dir, "test_x.npy"),
                    np.array(test_samples, dtype=np.float32))
            np.save(os.path.join(fold_dir, "test_y.npy"),
                    np.array(test_labels, dtype=np.int64))

            print(f"  Fold {fold_idx:02d} | test=S{test_subject:2d} | "
                  f"train={len(train_samples):4d} samples | "
                  f"test={len(test_samples):3d} samples | "
                  f"label dist={np.bincount(test_labels).tolist()}")

        print(f"\n{'=' * 60}")
        print(f"✅ LoSo 预处理完成！共 {len(self.subject_ids)} 折")
        print(f"数据已保存到: {self.output_path}")
        print(f"{'=' * 60}\n")