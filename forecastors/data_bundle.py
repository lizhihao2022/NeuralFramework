# forecastors/data_bundle.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import copy
import torch
import torch.utils.data as data

from datasets import DATASET_REGISTRY


@dataclass
class DataBundle:
    data_name: str
    data_args: Dict[str, Any]
    shape: List[int]

    train_loader: data.DataLoader
    valid_loader: data.DataLoader
    test_loader: data.DataLoader

    x_normalizer: Any
    y_normalizer: Any

    coords: Optional[torch.Tensor] = None
    geom: Optional[Dict[str, Any]] = None
    cache_path: Optional[str] = None


def build_data_bundle(
    *,
    data_name: str,
    data_args: Dict[str, Any],
    drop_last: bool = False,
    override_num_workers: Optional[int] = 0,
    override_pin_memory: Optional[bool] = False,
    override_train_batchsize: Optional[int] = None,
    override_eval_batchsize: Optional[int] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
) -> DataBundle:
    """
    Build dataset & loaders ONCE, to be shared across multiple forecasters/models.

    Recommended in notebooks:
      override_num_workers=0 to avoid worker process memory duplication.
    """
    if data_name not in DATASET_REGISTRY:
        raise NotImplementedError(f"Dataset {data_name} not implemented")

    dataset_kwargs = dataset_kwargs or {}

    # copy and override loader-related args to avoid surprises
    da = copy.deepcopy(data_args)
    if override_num_workers is not None:
        da["num_workers"] = int(override_num_workers)
    if override_pin_memory is not None:
        da["pin_memory"] = bool(override_pin_memory)
    if override_train_batchsize is not None:
        da["train_batchsize"] = int(override_train_batchsize)
    if override_eval_batchsize is not None:
        da["eval_batchsize"] = int(override_eval_batchsize)

    dataset_cls = DATASET_REGISTRY[data_name]
    dataset = dataset_cls(da, **dataset_kwargs)

    train_loader, valid_loader, test_loader, _ = dataset.make_loaders(
        ddp=False,
        rank=0,
        world_size=1,
        drop_last=drop_last,
    )

    shape = list(da.get("shape", []))
    return DataBundle(
        data_name=data_name,
        data_args=da,
        shape=shape,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        x_normalizer=getattr(dataset, "x_normalizer", None),
        y_normalizer=getattr(dataset, "y_normalizer", None),
        coords=getattr(dataset, "coords", None),
        geom=getattr(dataset, "geom", None),
        cache_path=getattr(dataset, "cache_path", None),
    )
