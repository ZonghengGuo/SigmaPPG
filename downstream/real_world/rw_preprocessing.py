import os
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from tqdm import tqdm
import pandas as pd


class PreprocessHumanID:
    """
    人体识别数据预处理类
    支持多种Excel格式
    """

    def __init__(self, args):
        self.args = args
        self.raw_data_path = args.raw_data_path
        self.save_path = os.path.join(args.seg_save_path, "humanid_processed")

        # 数据参数
        self.SAMPLING_RATE = 50  # Hz
        self.SIGNAL_LENGTH = 300  # 采样点数（6秒 × 50Hz）
        self.DURATION = 6  # 秒

        # 滤波器参数
        self.FILTER_LOWCUT = 0.5  # Hz，去除基线漂移
        self.FILTER_HIGHCUT = 8.0  # Hz，保留心率相关成分
        self.FILTER_ORDER = 4

        print(f"Human Identification Preprocessor Config:")
        print(f"  > Sampling Rate: {self.SAMPLING_RATE} Hz")
        print(f"  > Signal Length: {self.SIGNAL_LENGTH} samples ({self.DURATION}s)")
        print(f"  > Bandpass Filter: {self.FILTER_LOWCUT}-{self.FILTER_HIGHCUT} Hz")

    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        """设计带通滤波器"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def normalize_signal(self, sig):
        sig_min = np.min(sig)
        sig_max = np.max(sig)
        if sig_max - sig_min < 1e-9:
            return np.zeros_like(sig)
        return (sig - sig_min) / (sig_max - sig_min)

    def apply_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        """应用带通滤波器"""
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    def interpolate_signal(self, sig, target_length):
        """插值到目标长度"""
        current_length = len(sig)
        if current_length == target_length:
            return sig

        x_old = np.linspace(0, 1, current_length)
        x_new = np.linspace(0, 1, target_length)

        f = interp1d(x_old, sig, kind='cubic', fill_value='extrapolate')
        sig_interp = f(x_new)

        return sig_interp

    def process_single_signal(self, sig):
        """处理单个PPG信号"""
        # 1. 确保长度正确
        if len(sig) != self.SIGNAL_LENGTH:
            sig = self.interpolate_signal(sig, self.SIGNAL_LENGTH)

        # 2. 去除异常值
        median = np.median(sig)
        mad = np.median(np.abs(sig - median))
        threshold = 5 * mad
        sig = np.clip(sig, median - threshold, median + threshold)

        # 3. 带通滤波
        try:
            sig = self.apply_bandpass_filter(
                sig, self.FILTER_LOWCUT, self.FILTER_HIGHCUT,
                self.SAMPLING_RATE, self.FILTER_ORDER
            )
        except Exception as e:
            print(f"Warning: Filtering failed, using original signal. Error: {e}")

        # 4. 标准化
        sig = self.normalize_signal(sig)

        return sig

    def detect_excel_format(self, df):
        """
        自动检测Excel格式

        返回:
            format_type: 'two_row_metadata' 或 'single_row_metadata'
            id_row_idx: ID数据所在行的索引
            serial_row_idx: Serial行索引（如果有）
            skip_first_col: 是否跳过第一列
        """
        print("\n  🔍 Auto-detecting Excel format...")

        # 检查最后几行
        id_label_row_idx = None
        serial_label_row_idx = None

        for idx in range(len(df)):
            row_first_col = str(df.iloc[idx, 0]).strip().lower()

            if row_first_col == 'id':
                id_label_row_idx = idx
                print(f"     Found 'ID' label row at index {id_label_row_idx}")

            if row_first_col == 'serial':
                serial_label_row_idx = idx
                print(f"     Found 'Serial' label row at index {serial_label_row_idx}")

        if id_label_row_idx is not None:
            # 格式1: 有ID标签行，值在下一行
            if id_label_row_idx + 1 < len(df):
                print(f"     ✅ Detected format: TWO_ROW_METADATA")
                print(f"        ID label at row {id_label_row_idx}, values at row {id_label_row_idx + 1}")

                # 🔧 修复：检查ID值行的第0列是否是有效的ID值
                id_value_row = id_label_row_idx + 1
                first_col_val = df.iloc[id_value_row, 0]

                # 判断第0列是否是有效的ID值
                skip_first = True  # 默认跳过
                try:
                    first_col_num = float(first_col_val)
                    if not np.isnan(first_col_num) and 1 <= first_col_num <= 50:
                        print(f"        First column of ID row ({first_col_num}) is a valid ID - NOT skipping")
                        skip_first = False  # 第0列是有效ID，不跳过
                    else:
                        print(f"        First column of ID row ({first_col_num}) is not a valid ID - skipping")
                except:
                    print(f"        First column of ID row is not numeric - skipping")

                return 'two_row_metadata', id_value_row, serial_label_row_idx, skip_first
            else:
                raise ValueError(f"ID label found at row {id_label_row_idx}, but no next row for values!")
        else:
            # 格式2: 没有ID标签行，最后一行直接是ID值
            print(f"     ✅ Detected format: SINGLE_ROW_METADATA")

            # 检查最后一行
            last_row_idx = len(df) - 1
            print(f"        Checking last row (index {last_row_idx})...")

            # 检查最后一行第一列的值
            first_col_val = df.iloc[last_row_idx, 0]
            print(f"        First column value: {first_col_val}")

            # ✅ 修复：判断第一列是否是有效的ID值
            try:
                first_col_num = float(first_col_val)
                if not np.isnan(first_col_num) and 1 <= first_col_num <= 50:
                    print(f"        First column ({first_col_num}) looks like a valid subject ID")
                    skip_first = False  # 不跳过第一列
                else:
                    print(f"        First column ({first_col_num}) doesn't look like subject ID")
                    skip_first = True  # 跳过第一列
            except:
                print(f"        First column is not numeric, will skip it")
                skip_first = True

            # 检查整行的数值范围
            last_row_data = df.iloc[last_row_idx, :].values
            try:
                numeric_vals = [float(x) for x in last_row_data if not pd.isna(x)]
                if len(numeric_vals) > 0:
                    min_val = min(numeric_vals)
                    max_val = max(numeric_vals)
                    print(f"        Last row numeric range: {min_val} - {max_val}")

                    if 0 < min_val <= 50 and 0 < max_val <= 50:
                        print(f"        ✅ Values are in valid subject ID range (1-50)")
                        return 'single_row_metadata', last_row_idx, serial_label_row_idx, skip_first
                    else:
                        print(f"        ⚠️  Values seem out of range for subject IDs")
            except:
                pass

            # 如果找到Serial行，假设ID在Serial行之后
            if serial_label_row_idx is not None and serial_label_row_idx + 2 < len(df):
                print(f"        Trying: ID values at row {serial_label_row_idx + 2}")
                return 'single_row_metadata', serial_label_row_idx + 2, serial_label_row_idx, skip_first

            # 默认使用最后一行
            return 'single_row_metadata', last_row_idx, serial_label_row_idx, skip_first

    def load_from_excel(self, xlsx_path, sheet_name=0):
        """
        从Excel文件加载数据（支持多种格式）
        """
        print(f"Loading data from Excel: {xlsx_path}")

        # 读取Excel文件
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
        print(f"  Raw data shape: {df.shape}")

        # 自动检测格式
        format_type, id_value_row_idx, serial_label_row_idx, skip_first_col = self.detect_excel_format(df)

        # 提取ID值
        print(f"\n  📊 Extracting subject IDs from row {id_value_row_idx}...")

        # ✅ 修复：根据格式决定是否跳过第一列
        if skip_first_col:
            print(f"     Skipping first column (contains label or non-ID value)")
            subject_ids_raw = df.iloc[id_value_row_idx, 1:].values
            col_offset = 1
        else:
            print(f"     Including first column (contains valid ID value)")
            subject_ids_raw = df.iloc[id_value_row_idx, :].values
            col_offset = 0

        print(f"     First 10 values: {subject_ids_raw[:10]}")
        print(f"     Last 10 values: {subject_ids_raw[-10:]}")

        # 过滤有效的ID
        valid_cols = []
        valid_subject_ids = []

        for i, subj_id in enumerate(subject_ids_raw):
            col_idx = i + col_offset
            try:
                subj_id_num = float(subj_id)
                if not np.isnan(subj_id_num):
                    valid_cols.append(col_idx)
                    valid_subject_ids.append(int(subj_id_num))
            except (ValueError, TypeError):
                continue

        if len(valid_subject_ids) == 0:
            raise ValueError(f"No valid subject IDs found in row {id_value_row_idx}!")

        print(f"\n  ✅ Extraction Results:")
        print(f"     Valid columns: {len(valid_cols)}")
        print(f"     Subject IDs (first 20): {valid_subject_ids[:20]}")
        print(f"     Subject IDs (last 20): {valid_subject_ids[-20:]}")

        unique_subjects = sorted(set(valid_subject_ids))
        print(f"     Unique subjects: {unique_subjects}")
        print(f"     Number of unique subjects: {len(unique_subjects)}")

        # 验证
        min_id = min(valid_subject_ids)
        max_id = max(valid_subject_ids)
        print(f"     Subject ID range: {min_id} - {max_id}")

        # 确定信号数据范围
        if format_type == 'two_row_metadata':
            signal_end_row = id_value_row_idx - 1
            if serial_label_row_idx is not None and serial_label_row_idx < signal_end_row:
                signal_end_row = serial_label_row_idx
        else:
            signal_end_row = id_value_row_idx
            if serial_label_row_idx is not None and serial_label_row_idx < signal_end_row:
                signal_end_row = serial_label_row_idx

        # 跳过空行和序列号行
        while signal_end_row > 0:
            row_data = df.iloc[signal_end_row - 1, valid_cols].values

            # 检查是否是空行
            if pd.isna(row_data).all():
                signal_end_row -= 1
                continue

            # 🔧 修复：检查是否是序列号行（连续递增的整数）
            try:
                # 取前10个值检查是否是连续递增
                sample_vals = []
                for val in row_data[:min(10, len(row_data))]:
                    if not pd.isna(val):
                        sample_vals.append(float(val))

                if len(sample_vals) >= 3:
                    # 检查是否是连续递增的整数序列
                    is_serial = True
                    for i in range(len(sample_vals) - 1):
                        if abs(sample_vals[i + 1] - sample_vals[i] - 1.0) > 0.01:
                            is_serial = False
                            break

                    # 检查第一个值是否接近1
                    if is_serial and abs(sample_vals[0] - 1.0) < 0.01:
                        print(f"     Detected serial number row at index {signal_end_row - 1}, skipping...")
                        signal_end_row -= 1
                        continue
            except:
                pass

            # 如果不是空行也不是序列号行，停止
            break

        print(f"\n  📏 Signal data range: rows 0 to {signal_end_row}")

        # 提取信号数据
        signals = df.iloc[:signal_end_row, valid_cols].values.T
        print(f"     Extracted signals shape: {signals.shape}")

        # 转换为0-based标签
        subject_to_idx = {subj: idx for idx, subj in enumerate(unique_subjects)}
        y = np.array([subject_to_idx[subj] for subj in valid_subject_ids])

        print(f"\n  🏷️  Label Mapping (first 10, last 10):")
        for i, orig_id in enumerate(unique_subjects):
            if i < 10 or i >= len(unique_subjects) - 10:
                count = valid_subject_ids.count(orig_id)
                print(f"     Original ID {orig_id:2d} → Label {subject_to_idx[orig_id]:2d} ({count} samples)")
            elif i == 10:
                print(f"     ...")

        # 最终验证
        print(f"\n  ✅ Final Summary:")
        print(f"     Samples: {len(signals)}")
        print(f"     Subjects: {len(unique_subjects)}")
        print(f"     Signal shape: {signals.shape}")
        print(f"     Label range: {y.min()} - {y.max()}")

        assert y.min() >= 0, f"Negative labels: {y.min()}"
        assert y.max() < len(unique_subjects), f"Label out of range: {y.max()} >= {len(unique_subjects)}"

        return signals, y, unique_subjects

    def load_from_multiple_files(self, data_dir):
        """从多个文件加载数据"""
        xlsx_files = sorted([f for f in os.listdir(data_dir)
                             if f.endswith('.xlsx') or f.endswith('.xls')])

        if len(xlsx_files) == 0:
            raise ValueError(f"No Excel files found in {data_dir}")

        print(f"Found {len(xlsx_files)} Excel files: {xlsx_files}")

        all_X = []
        all_y = []
        all_subject_ids = []

        for xlsx_file in xlsx_files:
            xlsx_path = os.path.join(data_dir, xlsx_file)
            print(f"\n{'=' * 60}")
            print(f"Processing: {xlsx_file}")
            print(f"{'=' * 60}")

            try:
                X, y, subjects = self.load_from_excel(xlsx_path)

                original_ids = [subjects[label] for label in y]

                all_X.append(X)
                all_y.append(y)
                all_subject_ids.extend(original_ids)

                print(f"✅ Successfully loaded {len(X)} samples from {xlsx_file}")

            except Exception as e:
                print(f"❌ Error loading {xlsx_file}: {e}")
                print(f"   Skipping this file...")
                import traceback
                traceback.print_exc()
                continue

        if len(all_X) == 0:
            raise ValueError("No data loaded from any files!")

        # 合并数据
        X_combined = np.vstack(all_X)

        # 统一subject mapping
        unique_all_subjects = sorted(set(all_subject_ids))
        global_subject_to_idx = {subj: idx for idx, subj in enumerate(unique_all_subjects)}
        y_combined = np.array([global_subject_to_idx[subj] for subj in all_subject_ids])

        print(f"\n{'=' * 60}")
        print(f"📦 Combined Data Summary:")
        print(f"{'=' * 60}")
        print(f"  Total samples: {len(X_combined)}")
        print(f"  Total unique subjects: {len(unique_all_subjects)}")
        print(f"  Subject IDs: {unique_all_subjects}")
        print(f"  Label range: {y_combined.min()} - {y_combined.max()}")

        return X_combined, y_combined

    def preprocess_and_save(self):
        """主预处理流程"""
        os.makedirs(self.save_path, exist_ok=True)

        print(f"\n{'=' * 60}")
        print("Starting Human Identification Data Preprocessing")
        print(f"{'=' * 60}\n")

        # 加载数据
        print("Step 1: Loading raw data...")
        try:
            X_raw, y_raw = self.load_from_multiple_files(self.raw_data_path)
        except Exception as e:
            print(f"Error loading from multiple files: {e}")
            print("Trying single file...")
            xlsx_files = [f for f in os.listdir(self.raw_data_path)
                          if f.endswith('.xlsx') or f.endswith('.xls')]
            if len(xlsx_files) > 0:
                X_raw, y_raw, _ = self.load_from_excel(
                    os.path.join(self.raw_data_path, xlsx_files[0])
                )
            else:
                raise ValueError("No data files found!")

        print(f"\n  ✅ Data Loading Complete:")
        print(f"     Samples: {len(X_raw)}")
        print(f"     Subjects: {len(np.unique(y_raw))}")
        print(f"     Label range: {y_raw.min()} - {y_raw.max()}")

        # 处理信号
        print("\nStep 2: Processing signals...")
        X_processed = []

        for i in tqdm(range(len(X_raw)), desc="Processing"):
            sig = X_raw[i]
            sig_processed = self.process_single_signal(sig)
            X_processed.append(sig_processed)

        X_processed = np.array(X_processed)
        X_processed = X_processed[:, np.newaxis, :]

        print(f"\nStep 3: Saving processed data...")

        # 按受试者保存
        unique_subjects = np.unique(y_raw)

        for subject_idx in unique_subjects:
            mask = y_raw == subject_idx
            X_subject = X_processed[mask]
            y_subject = y_raw[mask]

            subject_str = f"S{subject_idx + 1:02d}"

            np.save(
                os.path.join(self.save_path, f"{subject_str}_x.npy"),
                X_subject.astype(np.float32)
            )
            np.save(
                os.path.join(self.save_path, f"{subject_str}_y.npy"),
                y_subject.astype(np.int64)
            )

            print(f"  {subject_str}: {len(X_subject)} samples")

        # 保存整体数据
        np.save(
            os.path.join(self.save_path, "all_x.npy"),
            X_processed.astype(np.float32)
        )
        np.save(
            os.path.join(self.save_path, "all_y.npy"),
            y_raw.astype(np.int64)
        )

        print(f"\n{'=' * 60}")
        print("✅ Preprocessing Complete!")
        print(f"{'=' * 60}")
        print(f"  Total samples: {len(X_processed)}")
        print(f"  Total subjects: {len(unique_subjects)}")
        print(f"  Signal shape: {X_processed.shape}")
        print(f"  Label range: {y_raw.min()} - {y_raw.max()}")
        print(f"  Saved to: {self.save_path}")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, required=True)
    parser.add_argument('--seg_save_path', type=str, default='./data')

    args = parser.parse_args()

    preprocessor = PreprocessHumanID(args)
    preprocessor.preprocess_and_save()