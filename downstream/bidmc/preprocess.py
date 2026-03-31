import os
import scipy.io
import numpy as np
from tqdm import tqdm
from scipy.signal import resample


class PreprocessBIDMC:
    def __init__(self, args):
        self.args = args
        self.dataset_path = args.raw_data_path
        self.save_path = os.path.join(args.seg_save_path, "bidmc_processed")

        self.TARGET_FS = args.rsfreq

        self.WINDOW_SEC = 32
        self.SHIFT_SEC = 4

        self.TARGET_LEN = int(self.WINDOW_SEC * self.TARGET_FS)
        self.SHIFT_LEN = int(self.SHIFT_SEC * self.TARGET_FS)

        print(f"BIDMC Preprocessor Config:")
        print(f"  > Target Fs: {self.TARGET_FS}Hz")
        print(f"  > Window: {self.WINDOW_SEC}s ({self.TARGET_LEN} points)")
        print(f"  > Shift: {self.SHIFT_SEC}s")

    def load_bidmc_subject_data(self, filename, subject_idx):
        """
        复现 PDF 中的鲁棒读取逻辑，并提取 HR 和 SpO2 参数
        """
        if not os.path.exists(filename):
            return None, None, None, None, None

        try:
            mat = scipy.io.loadmat(filename)
            if 'data' not in mat:
                raise ValueError("Mat file does not contain 'data' key.")

            subject_data = mat['data'][0, subject_idx]

            ppg_struct = subject_data['ppg']
            ppg_signal = ppg_struct[0, 0]['v'].flatten()
            fs_orig = int(ppg_struct[0, 0]['fs'][0, 0])

            ref_breaths = None
            ref_struct = None
            if 'ref' in subject_data.dtype.names:
                ref_struct = subject_data['ref']

            if ref_struct is not None and 'breaths' in ref_struct.dtype.names:
                raw_breaths = ref_struct[0, 0]['breaths']
                stack = [raw_breaths]
                visited = 0
                while stack and visited < 2000:
                    obj = stack.pop()
                    visited += 1
                    if (isinstance(obj, np.ndarray) and
                            obj.ndim >= 1 and
                            np.issubdtype(obj.dtype, np.number) and
                            obj.size > 10):
                        ref_breaths = obj.flatten()
                        break
                    if isinstance(obj, np.ndarray):
                        if obj.dtype == 'O':
                            for item in obj.flatten()[::-1]: stack.append(item)
                        elif obj.ndim == 0:
                            stack.append(obj.item())
                        elif obj.dtype.names:
                            for name in obj.dtype.names: stack.append(obj[name])
                    elif isinstance(obj, (list, tuple)):
                        for item in reversed(obj): stack.append(item)

            hr_params = None
            spo2_params = None

            if ref_struct is not None and 'params' in ref_struct.dtype.names:
                params_struct = ref_struct[0, 0]['params']

                if 'hr' in params_struct.dtype.names:
                    hr_raw = params_struct[0, 0]['hr']
                    stack = [hr_raw]
                    visited = 0
                    while stack and visited < 2000:
                        obj = stack.pop()
                        visited += 1
                        if (isinstance(obj, np.ndarray) and
                                obj.ndim >= 1 and
                                np.issubdtype(obj.dtype, np.number) and
                                obj.size > 10):
                            hr_params = obj.flatten()
                            break
                        if isinstance(obj, np.ndarray):
                            if obj.dtype == 'O':
                                for item in obj.flatten()[::-1]:
                                    stack.append(item)
                            elif obj.ndim == 0:
                                stack.append(obj.item())
                            elif obj.dtype.names:
                                for name in obj.dtype.names:
                                    stack.append(obj[name])
                        elif isinstance(obj, (list, tuple)):
                            for item in reversed(obj):
                                stack.append(item)

                if 'spo2' in params_struct.dtype.names:
                    spo2_raw = params_struct[0, 0]['spo2']
                    stack = [spo2_raw]
                    visited = 0
                    while stack and visited < 2000:
                        obj = stack.pop()
                        visited += 1
                        if (isinstance(obj, np.ndarray) and
                                obj.ndim >= 1 and
                                np.issubdtype(obj.dtype, np.number) and
                                obj.size > 10):
                            spo2_params = obj.flatten()
                            break
                        if isinstance(obj, np.ndarray):
                            if obj.dtype == 'O':
                                for item in obj.flatten()[::-1]:
                                    stack.append(item)
                            elif obj.ndim == 0:
                                stack.append(obj.item())
                            elif obj.dtype.names:
                                for name in obj.dtype.names:
                                    stack.append(obj[name])
                        elif isinstance(obj, (list, tuple)):
                            for item in reversed(obj):
                                stack.append(item)

            if ref_breaths is None:
                return None, None, None, None, None

            return ppg_signal, fs_orig, ref_breaths, hr_params, spo2_params

        except Exception as e:
            print(f"  Error reading subject {subject_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None

    def process_subject(self, ppg, fs_orig, ref_breaths, hr_params, spo2_params):
        total_sec = len(ppg) / fs_orig
        target_total_len = int(total_sec * self.TARGET_FS)
        ppg_resampled = resample(ppg, target_total_len)

        scale_factor = self.TARGET_FS / fs_orig
        ref_breaths_scaled = ref_breaths * scale_factor

        if hr_params is not None:
            hr_params = hr_params[:int(total_sec)]
        if spo2_params is not None:
            spo2_params = spo2_params[:int(total_sec)]

        samples = []
        labels_rr = []
        labels_hr = []
        labels_spo2 = []

        num_windows = (len(ppg_resampled) - self.TARGET_LEN) // self.SHIFT_LEN + 1

        for i in range(num_windows):
            start_idx = i * self.SHIFT_LEN
            end_idx = start_idx + self.TARGET_LEN

            x_window = ppg_resampled[start_idx:end_idx]

            # 计算窗口对应的时间范围（秒）
            start_time_sec = start_idx / self.TARGET_FS
            end_time_sec = end_idx / self.TARGET_FS

            # 找到落在窗口内的呼吸标注点
            breaths_in_window = ref_breaths_scaled[
                (ref_breaths_scaled >= start_idx) & (ref_breaths_scaled < end_idx)
                ]

            # 过滤逻辑: >3个呼吸才算有效
            if len(breaths_in_window) > 3:
                # 计算呼吸率标签
                intervals = np.diff(breaths_in_window)
                avg_interval_samples = np.mean(intervals)
                avg_interval_sec = avg_interval_samples / self.TARGET_FS

                if avg_interval_sec > 0:
                    y_rr = 60.0 / avg_interval_sec

                    if 5 <= y_rr <= 50:
                        # 计算心率标签（窗口内平均）
                        y_hr = None
                        if hr_params is not None and len(hr_params) > 0:
                            start_param_idx = int(start_time_sec)
                            end_param_idx = int(end_time_sec)

                            # 确保索引不超出范围
                            if end_param_idx > len(hr_params):
                                end_param_idx = len(hr_params)

                            if start_param_idx < len(hr_params):
                                window_hr = hr_params[start_param_idx:end_param_idx]

                                # 确保是数值数组
                                if len(window_hr) > 0 and np.issubdtype(window_hr.dtype, np.number):
                                    # 过滤无效值（0或NaN）
                                    valid_hr = window_hr[(window_hr > 30) & (window_hr < 200) & (~np.isnan(window_hr))]
                                    if len(valid_hr) > 0:
                                        y_hr = np.mean(valid_hr)

                        # 计算血氧饱和度标签（窗口内平均）
                        y_spo2 = None
                        if spo2_params is not None and len(spo2_params) > 0:
                            start_param_idx = int(start_time_sec)
                            end_param_idx = int(end_time_sec)

                            # 确保索引不超出范围
                            if end_param_idx > len(spo2_params):
                                end_param_idx = len(spo2_params)

                            if start_param_idx < len(spo2_params):
                                window_spo2 = spo2_params[start_param_idx:end_param_idx]

                                # 确保是数值数组
                                if len(window_spo2) > 0 and np.issubdtype(window_spo2.dtype, np.number):
                                    # 过滤无效值（<70% 或 >100%）
                                    valid_spo2 = window_spo2[
                                        (window_spo2 >= 70) & (window_spo2 <= 100) & (~np.isnan(window_spo2))]
                                    if len(valid_spo2) > 0:
                                        y_spo2 = np.mean(valid_spo2)

                        if y_hr is not None and y_spo2 is not None:
                            x_min = np.min(x_window)
                            x_max = np.max(x_window)
                            if x_max - x_min < 1e-9:
                                x_window = np.zeros_like(x_window)
                            else:
                                x_window = (x_window - x_min) / (x_max - x_min)

                            samples.append(x_window)
                            labels_rr.append(y_rr)
                            labels_hr.append(y_hr)
                            labels_spo2.append(y_spo2)

        return (np.array(samples),
                np.array(labels_rr),
                np.array(labels_hr),
                np.array(labels_spo2))

    def preprocess_save(self):
        os.makedirs(self.save_path, exist_ok=True)

        mat_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.mat')]
        if not mat_files:
            print(f"No .mat files found in {self.dataset_path}")
            return

        mat_file_path = os.path.join(self.dataset_path, mat_files[0])
        print(f"Processing file: {mat_file_path}")

        # BIDMC 通常包含 53 个受试者
        for subject_idx in tqdm(range(53), desc="Processing Subjects"):
            ppg, fs, refs, hr, spo2 = self.load_bidmc_subject_data(mat_file_path, subject_idx)

            if ppg is not None:
                X, y_rr, y_hr, y_spo2 = self.process_subject(ppg, fs, refs, hr, spo2)

                if len(X) > 0:
                    # (N, Length) -> (N, 1, Length)
                    X = X[:, np.newaxis, :]
                    subj_str = f"S{subject_idx + 1:02d}"

                    # 保存信号
                    np.save(os.path.join(self.save_path, f"{subj_str}_x.npy"),
                            X.astype(np.float32))

                    # 保存三个标签
                    np.save(os.path.join(self.save_path, f"{subj_str}_y_rr.npy"),
                            y_rr.astype(np.float32))
                    np.save(os.path.join(self.save_path, f"{subj_str}_y_hr.npy"),
                            y_hr.astype(np.float32))
                    np.save(os.path.join(self.save_path, f"{subj_str}_y_spo2.npy"),
                            y_spo2.astype(np.float32))

                    print(f"  Subject {subj_str}: {len(X)} samples saved")

        print(f"Preprocessing done. Data saved to {self.save_path}")


# 使用示例和数据加载器
class MultiTaskDataLoader:
    """
    用于多任务学习的数据加载示例
    """

    def __init__(self, data_path):
        self.data_path = data_path

    def load_subject(self, subject_id):
        """
        加载单个受试者的数据和所有标签

        Returns:
            X: shape (N, 1, Length) - PPG信号
            y_rr: shape (N,) - 呼吸率标签
            y_hr: shape (N,) - 心率标签
            y_spo2: shape (N,) - 血氧饱和度标签
        """
        subj_str = f"S{subject_id:02d}"

        X = np.load(os.path.join(self.data_path, f"{subj_str}_x.npy"))
        y_rr = np.load(os.path.join(self.data_path, f"{subj_str}_y_rr.npy"))
        y_hr = np.load(os.path.join(self.data_path, f"{subj_str}_y_hr.npy"))
        y_spo2 = np.load(os.path.join(self.data_path, f"{subj_str}_y_spo2.npy"))

        return X, y_rr, y_hr, y_spo2

    def load_all_subjects(self, subject_ids):
        """
        加载多个受试者的数据
        """
        X_all = []
        y_rr_all = []
        y_hr_all = []
        y_spo2_all = []

        for subject_id in subject_ids:
            try:
                X, y_rr, y_hr, y_spo2 = self.load_subject(subject_id)
                X_all.append(X)
                y_rr_all.append(y_rr)
                y_hr_all.append(y_hr)
                y_spo2_all.append(y_spo2)
            except FileNotFoundError:
                print(f"Subject {subject_id:02d} not found, skipping...")

        return (np.concatenate(X_all, axis=0),
                np.concatenate(y_rr_all, axis=0),
                np.concatenate(y_hr_all, axis=0),
                np.concatenate(y_spo2_all, axis=0))