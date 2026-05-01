import h5py
import bisect
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
import numpy as np
import logging
import torch

list_path = List[Path]


class SingleShockDataset(Dataset):
    """
    Reads a single HDF5 file containing pre-segmented PPG signals.
    Supports lazy loading and in-memory caching.
    """

    def __init__(self, file_path: Path, window_size: int = 12000, stride_size: int = 1, start_percentage: float = 0,  # Changed from 3750 to 12000
                 end_percentage: float = 1, load_to_memory: bool = True):
        self.__file_path = file_path
        self.__expected_segment_length = window_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage
        self.load_to_memory = load_to_memory

        self.__file = None
        self.__dataset_key = 'signals'
        self.__length = 0
        self.__feature_size = [1, self.__expected_segment_length]

        self.cached_signals = None
        # Store multiple features
        self.cached_features = {}

        self._pre_read_length_and_cache()

    def _pre_read_length_and_cache(self):
        try:
            with h5py.File(str(self.__file_path), 'r') as f:
                if self.__dataset_key not in f:
                    logging.error(f"Dataset '{self.__dataset_key}' not found in {self.__file_path}")
                    return

                dataset = f[self.__dataset_key]
                num_segments, segment_length = dataset.shape

                if segment_length != self.__expected_segment_length:
                    logging.warning(
                        f"Segment length mismatch. Expected {self.__expected_segment_length}, found {segment_length}.")

                self.__length = num_segments

                if self.load_to_memory:
                    self.cached_signals = dataset[:]

                    # Load all available features to memory
                    # Check for new version feature keys, fallback to old logic if not present
                    feature_keys = ['feat_amp', 'feat_skew', 'feat_avg']

                    for key in feature_keys:
                        if key in f:
                            self.cached_features[key] = f[key][:]
                        else:
                            # If no new features, initialize with zeros to avoid errors
                            # Assume patch number is window_size // patch_size = 12000 // 100 = 120
                            patch_num = 120  # Changed from 30 to 120

        except Exception as e:
            logging.error(f"Failed to read {self.__file_path}: {e}")
            self.__length = 0

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.__length:
            raise IndexError(f"Index {idx} out of range")

        if self.load_to_memory and self.cached_signals is not None:
            segment_data = self.cached_signals[idx]
            segment_data = np.expand_dims(segment_data, axis=0).astype(np.float32)

            feat_stack = np.stack([
                self.cached_features['feat_amp'][idx],
                self.cached_features['feat_skew'][idx],
                self.cached_features['feat_avg'][idx]
            ], axis=0).astype(np.float32)

            return segment_data, feat_stack

        # Lazy Loading mode (if memory is not enough)
        if self.__file is None:
            try:
                self.__file = h5py.File(str(self.__file_path), 'r', swmr=True)
            except Exception as e:
                raise RuntimeError(f"Failed to open {self.__file_path}: {e}")

        try:
            segment_data = self.__file[self.__dataset_key][idx]
            segment_data = np.expand_dims(segment_data, axis=0).astype(np.float32)

            feature_keys = ['feat_amp', 'feat_skew', 'feat_avg']
            feats = []
            for key in feature_keys:
                if key in self.__file:
                    feats.append(self.__file[key][idx])
                else:
                    feats.append(np.zeros(120, dtype=np.float32))

            feat_stack = np.stack(feats, axis=0).astype(np.float32)

            return segment_data, feat_stack

        except Exception as e:
            logging.error(f"Error reading segment {idx}: {e}")
            if self.__file:
                self.__file.close()
                self.__file = None
            raise RuntimeError(f"Read failure in {self.__file_path}") from e

    def __del__(self):
        if self.__file:
            try:
                self.__file.close()
            except:
                pass

    def free(self) -> None:
        if self.__file:
            try:
                self.__file.close()
            except:
                pass
            self.__file = None
        self.cached_signals = None
        self.cached_features = {}

    def get_ch_names(self):
        return ['PPG']


class ShockDataset(Dataset):
    """Integrates multiple HDF5 files processed by SingleShockDataset."""

    def __init__(self, file_paths: list_path, window_size: int = 12000, stride_size: int = 1,  # Changed from 3750 to 12000
                 start_percentage: float = 0,
                 end_percentage: float = 1,
                 load_to_memory: bool = False):

        self.__file_paths = file_paths
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage
        self.load_to_memory = load_to_memory

        self.__datasets = []
        self.__length = 0
        self.__feature_size = None
        self.__dataset_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        valid_datasets = []
        logging.info(f"Initializing ShockDataset with {len(self.__file_paths)} file paths...")

        dataset_idx = 0
        for i, file_path in enumerate(self.__file_paths):
            dataset = SingleShockDataset(file_path, self.__window_size, self.__stride_size, self.__start_percentage,
                                         self.__end_percentage, load_to_memory=self.load_to_memory)

            if len(dataset) > 0:
                valid_datasets.append(dataset)
                self.__dataset_idxes.append(dataset_idx)
                dataset_idx += len(dataset)
            else:
                pass

        self.__datasets = valid_datasets
        self.__length = dataset_idx

        if not self.__datasets:
            logging.error("CRITICAL: No valid datasets loaded.")
            return

        logging.info(f"Successfully loaded {len(self.__datasets)} files. Total segments: {self.__length}")
        self.__feature_size = self.__datasets[0].feature_size

    @property
    def feature_size(self):
        return self.__feature_size if self.__feature_size is not None else [1, self.__window_size]

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.__length:
            raise IndexError(f"Index {idx} out of range")

        # Binary search to locate file
        dataset_idx = bisect.bisect_right(self.__dataset_idxes, idx) - 1
        item_idx = idx - self.__dataset_idxes[dataset_idx]

        return self.__datasets[dataset_idx][item_idx]

    def free(self) -> None:
        for dataset in self.__datasets:
            dataset.free()

    def get_ch_names(self):
        if not self.__datasets:
            return ['PPG']
        return self.__datasets[0].get_ch_names()