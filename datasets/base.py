# datasets/base_dataset.py

from __future__ import annotations

import os
import os.path as osp
from abc import ABC
from typing import Any, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class TensorPairDataset(Dataset):
    """
    Simple tensor pair dataset: (x, y).

    Args:
        x: input tensor of shape (N, ...)
        y: target tensor of shape (N, ...)
        mode: 'train' / 'valid' / 'test' (for logging or debugging only)
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, mode: str = "train") -> None:
        self.mode = mode
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class BaseDataset(ABC):
    """
    Generic base dataset for supervised learning.

    Responsibilities:
      - Parse common fields from data_args (data_path, ratios, subset, etc.).
      - load_or_process():
          * if cache exists -> load from .pt file
          * else:
              - load_raw_data()
              - split train/valid/test
              - process each split via process_split()
              - save cache
      - wrap (x, y) tensors into TensorPairDataset as train/valid/test.
      - expose:
          * self.train_dataset
          * self.valid_dataset
          * self.test_dataset
          * self.normalizer (if any, defined by subclasses)

    Subclasses SHOULD implement:
      - load_raw_data(self, **kwargs) -> Any
      - process_split(self, data_split, mode: str, normalizer=None)
         -> (x, y, normalizer)

    Subclasses MAY override:
      - get_cache_path(self) -> str
      - split_data(self, raw) -> (train_block, valid_block, test_block)
      - make_dataset(self, x, y, mode) -> Dataset
      - save_to_cache / load_from_cache if custom format is needed.
    """

    def __init__(self, data_args: Dict[str, Any], **kwargs: Any) -> None:
        self.data_args = data_args
        self.data_path: str = data_args.get("data_path", "")

        # Split ratios
        self.train_ratio: float = data_args.get("train_ratio", 0.8)
        self.valid_ratio: float = data_args.get("valid_ratio", 0.1)
        self.test_ratio: float = data_args.get("test_ratio", 0.1)

        # Optional subset for quick experiments
        self.subset: bool = data_args.get("subset", False)
        self.subset_ratio: float = data_args.get("subset_ratio", 0.1)

        # Control whether to force re-processing
        self.reprocess: bool = data_args.get("reprocess", False)

        # Cache path
        self.cache_path: str = self.get_cache_path()

        # Normalizer (subclass decides its type and content)
        self.x_normalizer: Optional[Any] = None
        self.y_normalizer: Optional[Any] = None

        # Load or process data
        (
            train_x,
            train_y,
            valid_x,
            valid_y,
            test_x,
            test_y,
            self.x_normalizer,
            self.y_normalizer,
        ) = self.load_or_process(**kwargs)

        # Apply subset if requested
        if self.subset:
            def _apply_subset(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                n = max(1, int(len(x) * self.subset_ratio))
                return x[:n], y[:n]

            train_x, train_y = _apply_subset(train_x, train_y)
            valid_x, valid_y = _apply_subset(valid_x, valid_y)
            test_x, test_y = _apply_subset(test_x, test_y)

        # Wrap into PyTorch Dataset objects
        self.train_dataset = self.make_dataset(train_x, train_y, mode="train")
        self.valid_dataset = self.make_dataset(valid_x, valid_y, mode="valid")
        self.test_dataset = self.make_dataset(test_x, test_y, mode="test")

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def get_cache_path(self) -> str:
        """
        Build the default cache path for processed data.

        Default: <root>_processed.pt
          e.g. /path/ns.mat -> /path/ns_processed.pt

        Subclasses can override this to encode additional info, e.g.:
          - sample_factor
          - resolution
        """
        if not self.data_path:
            return "dataset_processed.pt"
        root, _ = osp.splitext(self.data_path)
        return root + "_processed.pt"

    def load_raw_data(self, **kwargs: Any) -> Any:
        """
        Subclass MUST implement this.

        Example return types:
          - torch.Tensor of shape (N, ...)
          - list/tuple of samples
        """
        raise NotImplementedError("Subclasses must implement load_raw_data().")

    def split_data(self, raw: Any) -> Tuple[Any, Any, Any]:
        """
        Default split along the first dimension.

        Works for:
          - torch.Tensor with shape (N, ...)
          - numpy array with shape (N, ...)
          - sequence types (list, tuple) with len()

        Subclasses can override if they need a more complex split logic.
        """
        if isinstance(raw, (torch.Tensor,)):
            N = raw.shape[0]
        else:
            # assume __len__ is available
            N = len(raw)

        train_end = int(N * self.train_ratio)
        valid_end = int(N * (self.train_ratio + self.valid_ratio))

        if isinstance(raw, torch.Tensor):
            train_block = raw[:train_end]
            valid_block = raw[train_end:valid_end]
            test_block = raw[valid_end:]
        else:
            train_block = raw[:train_end]
            valid_block = raw[train_end:valid_end]
            test_block = raw[valid_end:]

        return train_block, valid_block, test_block

    def process_split(
        self,
        data_split: Any,
        mode: str,
        x_normalizer: Optional[Any] = None,
        y_normalizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any], Optional[Any]]:
        """
        Subclass SHOULD override this.

        Default behavior:
          - treat data_split as both x and y (identity mapping)
          - no normalization

        Returns:
            x: input tensor
            y: target tensor
            normalizer: possibly updated normalizer (for 'train' mode)
        """
        raise NotImplementedError("Subclasses must implement process_split().")

    def make_dataset(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: str = "train",
    ) -> TensorPairDataset:
        """
        Subclass can override this if it needs a custom Dataset implementation.
        """
        return TensorPairDataset(x, y, mode=mode)

    def save_to_cache(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        valid_x: torch.Tensor,
        valid_y: torch.Tensor,
        test_x: torch.Tensor,
        test_y: torch.Tensor,
        x_normalizer: Optional[Any],
        y_normalizer: Optional[Any],
    ) -> None:
        """
        Default cache format: a tuple of (train_x, train_y, valid_x, valid_y, test_x, test_y, normalizer).
        """
        os.makedirs(osp.dirname(self.cache_path), exist_ok=True)
        torch.save(
            (train_x, train_y, valid_x, valid_y, test_x, test_y, x_normalizer, y_normalizer),
            self.cache_path,
        )

    def load_from_cache(self) -> Tuple[torch.Tensor, ...]:
        """
        Reverse of save_to_cache; subclasses can override for custom formats.
        """
        return torch.load(self.cache_path)

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------
    def load_or_process(self, **kwargs: Any):
        """
        Main entry:
          - if cache exists and not reprocess -> load_from_cache
          - else:
             * raw = load_raw_data()
             * train/valid/test = split_data(raw)
             * process each via process_split()
             * save_to_cache()
        """
        if osp.exists(self.cache_path) and not self.reprocess:
            print("Loading processed data from", self.cache_path)
            return self.load_from_cache()

        print("Processing data from raw file...")
        raw = self.load_raw_data(**kwargs)
        train_block, valid_block, test_block = self.split_data(raw)

        # Process each split
        train_x, train_y, x_normalizer, y_normalizer = self.process_split(
            train_block, mode="train", x_normalizer=None, y_normalizer=None
        )
        valid_x, valid_y, _, _ = self.process_split(
            valid_block, mode="valid", x_normalizer=x_normalizer, y_normalizer=y_normalizer
        )
        test_x, test_y, _, _ = self.process_split(
            test_block, mode="test", x_normalizer=x_normalizer, y_normalizer=y_normalizer
        )

        self.save_to_cache(train_x, train_y, valid_x, valid_y, test_x, test_y, x_normalizer, y_normalizer)

        return train_x, train_y, valid_x, valid_y, test_x, test_y, x_normalizer, y_normalizer

    def make_loaders(
        self,
        *,
        ddp: bool = False,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DistributedSampler]]:
        """
        Create train/valid/test DataLoaders using self.data_config.

        Args:
            ddp: whether using DistributedDataParallel.
            rank: process rank in DDP.
            world_size: world size in DDP.
            drop_last: whether to drop last batch in training.

        Returns:
            train_loader, valid_loader, test_loader, x_normalizer, y_normalizer
        """
        train_bs = self.data_args.get("train_batchsize", 10)
        eval_bs = self.data_args.get("eval_batchsize", 10)
        num_workers = self.data_args.get("num_workers", 0)
        pin_memory = self.data_args.get("pin_memory", True)

        # --------- train loader: optional DistributedSampler ---------
        if ddp:
            train_sampler: Optional[DistributedSampler] = DistributedSampler(
                self.train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=drop_last,
            )
            shuffle_train = False
        else:
            train_sampler = None
            shuffle_train = True

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_bs,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

        # --------- valid / test loaders: no sampler, no shuffle ---------
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=eval_bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=eval_bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, valid_loader, test_loader, train_sampler
