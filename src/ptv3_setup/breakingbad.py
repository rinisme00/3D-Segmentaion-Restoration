"""
BreakingBad Classification Dataset for Pointcept (HDF5 direct reader)

Reads enriched HDF5 files directly — no format conversion needed.
Structure identical to FantasticBreaksClsDataset but registered
as a separate module for Breaking Bad data.

Place this file at:  src/ptv3/pointcept/datasets/breakingbad.py
Then add to         src/ptv3/pointcept/datasets/__init__.py:
    from .breakingbad import *
"""

import os
import copy
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class BreakingBadClsDataset(Dataset):
    """Binary classification dataset: complete (0) vs broken (1).

    Reads *_data_enriched.h5 produced by compute_geometric_features.py
    for the Breaking Bad dataset.
    """

    def __init__(
        self,
        split="train",
        data_root="data/breakingbad_classification",
        class_names=None,
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.class_names = class_names or ["complete", "broken"]
        self.transform = Compose(transform)
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.loop = loop if not test_mode else 1

        if test_mode and test_cfg is not None:
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        # ── Load HDF5 into memory ────────────────────────────────────────
        h5_name = "train_data_enriched.h5" if split == "train" else "test_data_enriched.h5"
        h5_path = os.path.join(data_root, h5_name)
        assert os.path.exists(h5_path), f"HDF5 not found: {h5_path}"

        with h5py.File(h5_path, "r") as f:
            self.points = f["data"][:].astype(np.float32)
            self.labels = f["label"][:].astype(np.int64)

        self.num_samples = self.points.shape[0]
        self.num_points = self.points.shape[1]
        self.num_dims = self.points.shape[2]

        logger = get_root_logger()
        logger.info(
            f"BreakingBadClsDataset [{split}]: "
            f"{self.num_samples} samples x {self.loop} loop, "
            f"{self.num_points} points, {self.num_dims}D "
            f"from {h5_path}"
        )

    def get_data(self, idx):
        data_idx = idx % self.num_samples
        sample = self.points[data_idx]
        coord = sample[:, :3].copy()
        feat = sample[:, 3:].copy()
        category = self.labels[data_idx].flatten().copy()
        return dict(coord=coord, feat=feat, category=category)

    def get_data_name(self, idx):
        data_idx = idx % self.num_samples
        return f"{self.split}_{data_idx:04d}"

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return self.num_samples * self.loop

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        assert idx < self.num_samples
        data_dict = self.get_data(idx)
        category = data_dict.pop("category")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(copy.deepcopy(data_dict)))
        for i in range(len(data_dict_list)):
            data_dict_list[i] = self.post_transform(data_dict_list[i])
        data_dict = dict(
            voting_list=data_dict_list,
            category=category,
            name=self.get_data_name(idx),
        )
        return data_dict
