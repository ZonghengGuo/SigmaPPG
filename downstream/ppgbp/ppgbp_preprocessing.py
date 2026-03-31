import os
import pandas as pd
import numpy as np
import zipfile
import requests
from tqdm import tqdm
from downstream.ppgbp.utils import resample_batch_signal, preprocess_one_ppg_signal
import joblib
from sklearn.model_selection import KFold



class PPGDataProcessor:
    def __init__(self, zippath: str, ppgpath: str, fs_target: int, n_folds: int = 5):
        self.zippath = zippath
        self.ppgpath = ppgpath
        self.fs_target = fs_target
        self.fs = 1000
        self.df = None
        self.n_folds = n_folds

    def downloadextract_PPGfiles(self, redownload: bool = False) -> None:
        """
        Downloads and extracts PPG files if they do not already exist or if redownload is requested.

        :param redownload: Flag to force re-download and extraction of PPG files.
        """
        if os.path.exists(self.ppgpath) and not redownload:
            print("PPG files already exist")
            return

        link = "https://figshare.com/ndownloader/articles/5459299/versions/5"
        print("Downloading PPG files (2.33 MB) ...")
        self.download_file(link, self.zippath)

        print("Unzipping PPG files ...")
        with zipfile.ZipFile(self.zippath, "r") as zip_ref:
            zip_ref.extractall(self.ppgpath)

        zip_path_main = os.path.join(self.ppgpath, "PPG-BP Database.zip")
        with zipfile.ZipFile(zip_path_main, "r") as zip_ref:
            zip_ref.extractall(self.ppgpath)

        os.remove(self.zippath)
        os.remove(zip_path_main)
        print("Done extracting and downloading")

    def download_file(self, url: str, filename: str) -> str:
        chunk_size = 1024
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(filename, "wb") as f, tqdm(unit="B", total=total) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
        return filename

    def process_data(self):
        self.df = pd.read_excel(f"{self.ppgpath}/PPG-BP dataset.xlsx", header=1)
        subjects = self.df.subject_ID.values
        main_dir = f"{self.ppgpath}/0_subject/"
        ppg_dir = f"{self.ppgpath}/ppg/"

        if not os.path.exists(ppg_dir):
            os.mkdir(ppg_dir)

        filenames = [f.split("_")[0] for f in os.listdir(main_dir)]

        for f in tqdm(filenames):
            segments = []
            for s in range(1, 4):
                print(f"Processing: {f}_{s}")
                signal = pd.read_csv(f"{main_dir}{f}_{str(s)}.txt", sep='\t', header=None)
                signal = signal.values.squeeze()[:-1]

                sig_min = np.min(signal)
                sig_max = np.max(signal)
                if sig_max - sig_min < 1e-9:
                    normalized_signal = np.zeros_like(signal)
                else:
                    normalized_signal = (signal - sig_min) / (sig_max - sig_min)
                signal, _, _, _ = preprocess_one_ppg_signal(waveform=normalized_signal, frequency=self.fs)
                resampled_signal = resample_batch_signal(signal, fs_original=self.fs, fs_target=self.fs_target, axis=0)

                padding_needed = 10 * self.fs_target - len(resampled_signal)
                pad_left = padding_needed // 2
                pad_right = padding_needed - pad_left

                padded_signal = np.pad(resampled_signal, pad_width=(pad_left, pad_right))
                segments.append(padded_signal)

            segments = np.vstack(segments)
            child_dir = f.zfill(4)
            self.save_segments_to_directory(ppg_dir, child_dir, segments)

        self.prepare_kfold_splits()
        self.save_kfold_data(ppg_dir)

    def save_segments_to_directory(self, save_dir: str, dir_name: str, segments: np.ndarray):
        subject_dir = os.path.join(save_dir, dir_name)
        os.makedirs(subject_dir, exist_ok=True)
        for i, segment in enumerate(segments):
            joblib.dump(segment, os.path.join(subject_dir, f'{i}.p'))

    def prepare_kfold_splits(self):
        """准备K折交叉验证的数据划分"""
        # 重命名列
        self.df = self.df.rename(columns={
            "Sex(M/F)": "sex",
            "Age(year)": "age",
            "Systolic Blood Pressure(mmHg)": "sysbp",
            "Diastolic Blood Pressure(mmHg)": "diasbp",
            "Heart Rate(b/m)": "hr",
            "BMI(kg/m^2)": "bmi"
        }).fillna(0)
        self.df['Hypertension_Code'] = pd.factorize(self.df['Hypertension'])[0]

        # 获取所有subject IDs
        all_subject_ids = self.df.subject_ID.values

        # 创建K折划分
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # 保存每折的划分信息
        fold_dir = os.path.join(self.ppgpath, 'folds')
        os.makedirs(fold_dir, exist_ok=True)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(all_subject_ids)):
            train_ids = all_subject_ids[train_idx]
            test_ids = all_subject_ids[test_idx]

            df_train = self.df[self.df.subject_ID.isin(train_ids)]
            df_test = self.df[self.df.subject_ID.isin(test_ids)]

            df_train.to_csv(f"{fold_dir}/fold{fold_idx}_train.csv", index=False)
            df_test.to_csv(f"{fold_dir}/fold{fold_idx}_test.csv", index=False)

            print(f"Fold {fold_idx}: Train={len(train_ids)}, Test={len(test_ids)}")

    def save_kfold_data(self, ppg_dir):
        """保存K折交叉验证的数据"""
        fold_dir = os.path.join(self.ppgpath, 'folds')

        for fold_idx in range(self.n_folds):
            # 读取当前fold的train和test IDs
            df_train = pd.read_csv(f"{fold_dir}/fold{fold_idx}_train.csv")
            df_test = pd.read_csv(f"{fold_dir}/fold{fold_idx}_test.csv")

            train_ids = df_train.subject_ID.values
            test_ids = df_test.subject_ID.values

            # 处理训练集
            train_data, train_sysbp, train_diasbp, train_hr, train_ht = [], [], [], [], []

            for id_i in train_ids:
                for j in range(3):
                    signal = joblib.load(os.path.join(ppg_dir, f"{id_i:04}", f'{j}.p'))[None, None, :]
                    train_data.append(signal)

                    row = self.df[self.df["subject_ID"] == id_i]
                    if row.empty:
                        print(f"No data found for subject_ID {id_i}")
                        continue

                    train_sysbp.append(row["sysbp"].values[0])
                    train_diasbp.append(row["diasbp"].values[0])
                    train_hr.append(row["hr"].values[0])
                    train_ht.append(row["Hypertension_Code"].values[0])

            # 处理测试集
            test_data, test_sysbp, test_diasbp, test_hr, test_ht = [], [], [], [], []

            for id_i in test_ids:
                for j in range(3):
                    signal = joblib.load(os.path.join(ppg_dir, f"{id_i:04}", f'{j}.p'))[None, None, :]
                    test_data.append(signal)

                    row = self.df[self.df["subject_ID"] == id_i]
                    if row.empty:
                        print(f"No data found for subject_ID {id_i}")
                        continue

                    test_sysbp.append(row["sysbp"].values[0])
                    test_diasbp.append(row["diasbp"].values[0])
                    test_hr.append(row["hr"].values[0])
                    test_ht.append(row["Hypertension_Code"].values[0])

            # 转换为numpy数组并保存
            train_data = np.concatenate(train_data)
            test_data = np.concatenate(test_data)

            np.save(os.path.join(fold_dir, f"fold{fold_idx}_train_X_ppg_{self.fs_target}Hz"), train_data)
            np.save(os.path.join(fold_dir, f"fold{fold_idx}_train_y_sysbp"), np.array(train_sysbp))
            np.save(os.path.join(fold_dir, f"fold{fold_idx}_train_y_diasbp"), np.array(train_diasbp))
            np.save(os.path.join(fold_dir, f"fold{fold_idx}_train_y_hr"), np.array(train_hr))
            np.save(os.path.join(fold_dir, f"fold{fold_idx}_train_y_ht"), np.array(train_ht))

            np.save(os.path.join(fold_dir, f"fold{fold_idx}_test_X_ppg_{self.fs_target}Hz"), test_data)
            np.save(os.path.join(fold_dir, f"fold{fold_idx}_test_y_sysbp"), np.array(test_sysbp))
            np.save(os.path.join(fold_dir, f"fold{fold_idx}_test_y_diasbp"), np.array(test_diasbp))
            np.save(os.path.join(fold_dir, f"fold{fold_idx}_test_y_hr"), np.array(test_hr))
            np.save(os.path.join(fold_dir, f"fold{fold_idx}_test_y_ht"), np.array(test_ht))

            print(f"Fold {fold_idx} saved: Train shape={train_data.shape}, Test shape={test_data.shape}")


def main(newhz: int, zippath: str, ppgpath: str, n_folds: int = 5):
    processor = PPGDataProcessor(zippath=zippath, ppgpath=ppgpath, fs_target=newhz, n_folds=n_folds)
    processor.downloadextract_PPGfiles()
    processor.process_data()
    print(f"PPGBP PPG data files for {n_folds}-fold cross-validation are ready in {os.path.abspath(processor.ppgpath)}")


class PreprocessPPGBP:
    """
    Adapter class to wrap PPGDataProcessor for compatibility with downstream_main.py
    """

    def __init__(self, args):
        self.args = args
        self.ppgpath = args.seg_save_path

        base_dir = os.path.dirname(args.seg_save_path)
        self.zippath = os.path.join(base_dir, "ppg_ppgbp.zip")

        self.fs_target = getattr(args, 'rsfreq', getattr(args, 'sampling_rate', 50))
        self.n_folds = getattr(args, 'n_folds', 5)

        self.processor = PPGDataProcessor(
            zippath=self.zippath,
            ppgpath=self.ppgpath,
            fs_target=self.fs_target,
            n_folds=self.n_folds
        )

    def preprocess_save(self):
        """
        Main preprocessing method that matches the interface of other downstream datasets
        """
        print(f"🚀 Starting PPGBP preprocessing with {self.n_folds}-fold cross-validation")
        print(f"Sampling rate: {self.fs_target} Hz")
        print(f"Zip file path: {self.zippath}")
        print(f"Data save path: {self.ppgpath}")

        self.processor.downloadextract_PPGfiles()
        self.processor.process_data()

        print(f"✅ PPGBP preprocessing completed!")
        print(f"Processed data saved to: {os.path.abspath(self.ppgpath)}")