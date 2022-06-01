from os import PathLike
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    data_h5_path: Path
    data_h5_keys: list[str]

    def __init__(self, data_h5_path: PathLike, _transform=None, _target_transform=None):
        self.data_h5_path = Path(data_h5_path)
        with h5py.File(self.data_h5_path) as fp:
            self.data_h5_keys = list(fp.keys())

    def __len__(self):
        return len(self.data_h5_keys)

    def __getitem__(self, index) -> tuple[torch.Tensor, ...]:
        if index >= len(self):
            raise IndexError()
        with h5py.File(self.data_h5_path) as fp:
            ds = fp[f'{index:06d}']
            array = np.asarray(ds)
            array = np.float32(np.expand_dims(array / np.max(array, axis=(1, 2)), axis=0))
            return torch.tensor(array), torch.tensor(ds.attrs['target'], dtype=torch.long)
