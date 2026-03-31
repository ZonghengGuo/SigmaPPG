import os
import numpy as np
import pandas as pd
import wfdb
import argparse
from scipy.signal import butter, resample, sosfiltfilt
from scipy.stats import skew
from tqdm import tqdm
from glob import glob
import logging
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import vitaldb
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseProcessor:
    def __init__(self, args: argparse.Namespace):
        self.raw_data_path = args.raw_data_path
        self.seg_save_path = args.seg_save_path
        self.target_sfreq = args.rsfreq
        self.dataset_name = args.dataset_name

    @staticmethod
    def filter_ppg_channel(data: np.ndarray, fs: int) -> np.ndarray:
        sos = butter(2, [0.5, 8], btype='band', fs=fs, output='sos')
        filtered_data = sosfiltfilt(sos, data)
        return filtered_data

    @staticmethod
    def resample_waveform(signal: np.ndarray, target_length: int) -> np.ndarray:
        resampled_signal = resample(signal, target_length)
        return resampled_signal

    @staticmethod
    def normalize_to_minus_one_to_one(data):
        if data.size == 0 or np.all(data == data[0]):
            return data
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == max_val:
            return np.zeros_like(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data

    @staticmethod
    def interpolate_nan(sig):
        interpolated_signal = pd.Series(sig).interpolate(method='linear', limit_direction='both').to_numpy()
        return interpolated_signal

    @staticmethod
    def is_constant_signal(signal):
        if signal.size == 0:
            return True
        return np.all(signal == signal[0])

    @staticmethod
    def is_nan_ratio_exceeded(signal, threshold):
        if signal.size == 0:
            return False
        nan_count = np.isnan(signal).sum()
        nan_ratio = nan_count / signal.size
        return nan_ratio > threshold

    @staticmethod
    def normalize_minmax(data):
        d_min = np.min(data)
        d_max = np.max(data)
        if d_max - d_min < 1e-9:
            return np.zeros_like(data)
        return (data - d_min) / (d_max - d_min)

    @staticmethod
    def calculate_amplitude_stability_sqi(signal, patch_size=125, min_valid_std=0.05, max_valid_std=2.0):
        num_patches = len(signal) // patch_size
        patch_stds = []

        for i in range(num_patches):
            start = i * patch_size
            end = (i + 1) * patch_size
            segment = signal[start:end]
            std_val = np.std(segment)
            patch_stds.append(std_val)

        patch_stds = np.array(patch_stds)

        median_std = np.median(patch_stds)
        mad = np.median(np.abs(patch_stds - median_std))

        if mad < 1e-9:
            rel_scores = np.ones(num_patches)
        else:
            modified_z = 0.6745 * (patch_stds - median_std) / mad
            rel_scores = np.exp(-0.2 * (modified_z ** 2))

        rise_k = 50
        score_low_gate = 1.0 / (1.0 + np.exp(-rise_k * (patch_stds - min_valid_std)))

        fall_k = 5
        score_high_gate = 1.0 / (1.0 + np.exp(fall_k * (patch_stds - max_valid_std)))

        abs_scores = score_low_gate * score_high_gate

        final_scores = rel_scores * abs_scores

        return final_scores

    @staticmethod
    def chunk_segments(slide_segment_time, original_fs, chunk_ppg_signal, nan_limit, target_fs,
                       patch_length_sec=1.0):
        valid_segments_from_chunk = []
        slide_segment_length = int(slide_segment_time * original_fs)

        local_total_attempts = 0
        local_kept_count = 0

        SKEW_MIN = 0.0
        SKEW_MAX = 1.8

        for start in range(0, len(chunk_ppg_signal) - slide_segment_length + 1, slide_segment_length):
            end = start + slide_segment_length
            slide_segment = chunk_ppg_signal[start:end]

            local_total_attempts += 1

            if np.all(np.isnan(slide_segment)):
                continue

            interpolated_segment = BaseProcessor.interpolate_nan(slide_segment)

            if BaseProcessor.is_nan_ratio_exceeded(interpolated_segment, nan_limit):
                continue

            if BaseProcessor.is_constant_signal(interpolated_segment):
                continue

            filtered_segment = BaseProcessor.filter_ppg_channel(interpolated_segment, original_fs)

            target_length = int(target_fs * slide_segment_time)
            current_length = len(filtered_segment)

            if current_length == target_length:
                resampled_segment = filtered_segment
            else:
                resampled_segment = BaseProcessor.resample_waveform(filtered_segment, target_length)

            normalized_segment = BaseProcessor.normalize_to_minus_one_to_one(resampled_segment)

            # if not skip_skewness_filter:
            #     segment_skew = skew(normalized_segment, bias=False)
            #
            #     if np.isnan(segment_skew):
            #         continue
            #
            #     if segment_skew < SKEW_MIN or segment_skew > SKEW_MAX:
            #         continue

            patch_length = int(patch_length_sec * target_fs)

            num_patches = len(normalized_segment) // patch_length

            if num_patches == 0 or patch_length == 0:
                continue

            norm_amp = BaseProcessor.calculate_amplitude_stability_sqi(
                normalized_segment,
                patch_size=patch_length,
                min_valid_std=0.05,
                max_valid_std=2.0
            )

            limit_len = num_patches * patch_length
            patches = normalized_segment[:limit_len].reshape(num_patches, patch_length)

            # Abs Skewness
            patch_skew = skew(patches, axis=1, bias=False)
            patch_skew = np.nan_to_num(patch_skew, nan=0.0)
            abs_skew = np.abs(patch_skew)
            norm_skew = np.tanh(abs_skew)

            # Fusion Score: 0.5 * norm_amp + 0.5 * norm_skew
            final_priority_scores = 0.5 * norm_amp + 0.5 * norm_skew

            valid_segments_from_chunk.append({
                "signal": normalized_segment,
                "avg": final_priority_scores,
                "amp": norm_amp,
                "skew": norm_skew
            })

            local_kept_count += 1

        return valid_segments_from_chunk, local_total_attempts, local_kept_count

    def _save_chunk_to_h5(self, segment_buffer_dicts, file_index):
        filename = f"{self.dataset_name}_segments_part_{file_index:04d}.h5"
        save_path = os.path.join(self.seg_save_path, filename)
        os.makedirs(self.seg_save_path, exist_ok=True)

        segment_buffer = [item["signal"] for item in segment_buffer_dicts]
        raw_amp_buffer = [item["amp"] for item in segment_buffer_dicts]
        raw_skew_buffer = [item["skew"] for item in segment_buffer_dicts]
        raw_avg_buffer = [item["avg"] for item in segment_buffer_dicts]

        try:
            with h5py.File(save_path, 'w') as hf:
                hf.create_dataset('signals', data=np.array(segment_buffer, dtype=np.float32))
                hf.create_dataset('feat_amp', data=np.array(raw_amp_buffer, dtype=np.float32))
                hf.create_dataset('feat_skew', data=np.array(raw_skew_buffer, dtype=np.float32))
                hf.create_dataset('feat_avg', data=np.array(raw_avg_buffer, dtype=np.float32))

            logging.info(f"✅ Saved {len(segment_buffer)} segments to {filename}")
        except Exception as e:
            logging.error(f"❌ Failed to save chunk to {save_path}: {e}", exc_info=True)


class MimicProcessor(BaseProcessor):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.processing_params = {
            "slide_segment_time": args.window_length,
            "nan_limit": 0.2,
            "req_seg_duration": args.window_length,
            "target_fs": self.target_sfreq,
            "patch_length_sec": args.patch_length
        }
        self.segments_per_file = args.h5_file_numbers

    def get_all_records(self):
        dataset_path = self.raw_data_path
        all_hea_files = glob(os.path.join(dataset_path, '**', '*.hea'), recursive=True)
        return [os.path.splitext(p)[0] for p in all_hea_files]

    @staticmethod
    def _worker_process_task(task_args):
        wave_path, start_sample, end_sample, ppg_index, original_fs, params = task_args

        try:
            chunk_record = wfdb.rdrecord(
                wave_path,
                sampfrom=start_sample,
                sampto=end_sample,
                channels=[ppg_index]
            )
            chunk_ppg_signal = chunk_record.p_signal.flatten()

            valid_segments, attempts, kept = BaseProcessor.chunk_segments(
                params['slide_segment_time'],
                original_fs,
                chunk_ppg_signal,
                params['nan_limit'],
                params['target_fs'],
                params['patch_length_sec']
            )

            return valid_segments, attempts, kept

        except FileNotFoundError:
            return None, 0, 0
        except Exception as e:
            logging.error(f"CRITICAL ERROR! 💥 Failed during processing of chunk from {wave_path}: {e}", exc_info=True)
            return None, 0, 0

    def run_processing(self):
        all_record_paths = self.get_all_records()
        all_record_paths.sort()

        tasks = []

        logging.info("Pre-processing: Scanning records...")
        for path in tqdm(all_record_paths, desc="Scanning records"):
            try:
                header = wfdb.rdheader(path)
                if header is None or header.sig_name is None: continue
                sig_names_lower = [s.lower() for s in header.sig_name]
                if "pleth" not in sig_names_lower: continue
                ppg_index = sig_names_lower.index("pleth")

                # Check for .dat file existence
                if header.file_name is not None:
                    pleth_file_name = header.file_name[ppg_index]
                    if pleth_file_name is not None and pleth_file_name != '~':
                        full_dat_path = os.path.join(os.path.dirname(path), pleth_file_name)
                        if not os.path.exists(full_dat_path): continue

                total_signal_length = header.sig_len
                original_fs = float(header.fs)
                if original_fs <= 0: continue
                if total_signal_length / original_fs < self.processing_params['req_seg_duration']: continue

                chunk_duration_seconds = 300
                chunk_length_samples = int(chunk_duration_seconds * original_fs)

                for start_sample in range(0, total_signal_length, chunk_length_samples):
                    end_sample = min(start_sample + chunk_length_samples, total_signal_length)
                    task_info = (path, start_sample, end_sample, ppg_index, original_fs, self.processing_params)
                    tasks.append(task_info)

            except Exception as e:
                logging.warning(f"Could not process header for {path}: {e}")

        logging.info(f"Scanning completed. Valid tasks created: {len(tasks)}")

        if not tasks:
            logging.warning("No valid tasks created.")
            return

        segment_buffer = []
        file_counter = 0
        num_workers = 8
        BATCH_SIZE = 20000

        total_segments_processed = 0
        total_segments_saved = 0

        logging.info(f"🚀 Starting Batch Processing...")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i in range(0, len(tasks), BATCH_SIZE):
                batch_tasks = tasks[i: i + BATCH_SIZE]
                futures = {executor.submit(MimicProcessor._worker_process_task, task): task for task in batch_tasks}

                for future in tqdm(as_completed(futures), total=len(batch_tasks), desc=f"Batch {i // BATCH_SIZE + 1}"):
                    try:
                        result_segments, n_attempts, n_kept = future.result(timeout=300)

                        total_segments_processed += n_attempts
                        total_segments_saved += n_kept

                        if result_segments:
                            segment_buffer.extend(result_segments)
                            while len(segment_buffer) >= self.segments_per_file:
                                chunk_to_save = segment_buffer[:self.segments_per_file]
                                self._save_chunk_to_h5(chunk_to_save, file_counter)
                                segment_buffer = segment_buffer[self.segments_per_file:]
                                file_counter += 1

                    except TimeoutError:
                        logging.error(f"TIMEOUT! Chunk processing took too long.")
                    except Exception as e:
                        logging.error(f"ERROR! {e}")

                del futures
                gc.collect()

        if segment_buffer:
            logging.info(f"Saving remaining {len(segment_buffer)} segments...")
            self._save_chunk_to_h5(segment_buffer, file_counter)

        logging.info("=" * 40)
        logging.info("🎉 DATA PROCESSING FINAL REPORT")
        logging.info("=" * 40)
        logging.info(f"Total Segments Analyzed : {total_segments_processed}")
        logging.info(f"Total Segments Kept     : {total_segments_saved}")

        if total_segments_processed > 0:
            retention_rate = (total_segments_saved / total_segments_processed) * 100
            logging.info(f"Retention Rate          : {retention_rate:.2f}%")
            logging.info(f"Discarded Rate          : {100 - retention_rate:.2f}%")
        else:
            logging.info("Retention Rate          : N/A (No segments processed)")
        logging.info("=" * 40)


class VitaldbProcessor(BaseProcessor):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.slide_segment_time = args.window_length
        self.nan_limit = 0.2
        self.target_fs = self.target_sfreq
        self.segments_per_file = args.h5_file_numbers
        self.patch_length_sec = args.patch_length

    def get_all_records(self):
        dataset_path = self.raw_data_path
        all_vital_files = glob(os.path.join(dataset_path, '*.vital'), recursive=True)
        return all_vital_files

    def process_record_chunk(self, wave_path):
        try:
            vf = vitaldb.VitalFile(wave_path)
            ppg = vf.to_numpy(['SNUADC/PLETH'], 1 / self.target_fs)
            original_fs = self.target_fs
            chunk_ppg_signal = ppg.flatten()

            valid_segments, attempts, kept = BaseProcessor.chunk_segments(
                self.slide_segment_time,
                original_fs,
                chunk_ppg_signal,
                self.nan_limit,
                self.target_fs,
                self.patch_length_sec
            )

            return valid_segments, attempts, kept

        except Exception as e:
            if 'SNUADC/PLETH' in str(e):
                logging.warning(f"Track 'SNUADC/PLETH' not found in {wave_path}. Skipping.")
            else:
                logging.error(f"CRITICAL ERROR! {e}")
            return None, 0, 0

    def run_processing(self):
        all_record_paths = self.get_all_records()
        all_record_paths.sort()

        if not all_record_paths:
            logging.warning("No .vital files found.")
            return

        tasks = all_record_paths
        logging.info(f"Total tasks created: {len(tasks)}")

        segment_buffer = []
        file_counter = 0
        num_workers = 8

        total_segments_processed = 0
        total_segments_saved = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.process_record_chunk, task): task for task in tasks}

            for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing VitalDB"):
                try:
                    result_segments, n_attempts, n_kept = future.result(timeout=300)

                    total_segments_processed += n_attempts
                    total_segments_saved += n_kept

                    if result_segments:
                        segment_buffer.extend(result_segments)
                        while len(segment_buffer) >= self.segments_per_file:
                            chunk_to_save = segment_buffer[:self.segments_per_file]
                            self._save_chunk_to_h5(chunk_to_save, file_counter)
                            segment_buffer = segment_buffer[self.segments_per_file:]
                            file_counter += 1
                except TimeoutError:
                    logging.error(f"TIMEOUT!")
                except Exception as e:
                    logging.error(f"ERROR! {e}")

        if segment_buffer:
            self._save_chunk_to_h5(segment_buffer, file_counter)

        logging.info("=" * 40)
        logging.info("🎉 DATA PROCESSING FINAL REPORT (VitalDB)")
        logging.info("=" * 40)
        logging.info(f"Total Segments Analyzed : {total_segments_processed}")
        logging.info(f"Total Segments Kept     : {total_segments_saved}")

        if total_segments_processed > 0:
            retention_rate = (total_segments_saved / total_segments_processed) * 100
            logging.info(f"Retention Rate          : {retention_rate:.2f}%")
            logging.info(f"Discarded Rate          : {100 - retention_rate:.2f}%")
        logging.info("=" * 40)