# forecastors/base.py
import os
from typing import Any, Dict, Optional

import yaml
import torch
import torch.utils.data as data

from models import MODEL_REGISTRY
from datasets import DATASET_REGISTRY
from utils import Evaluator, set_seed


class BaseForecaster(object):
    """
    Lightweight inference / evaluation helper.

    Given a training run directory:
      - load config.yaml
      - build the model
      - load best_model.pth
      - provide forecast / visualization utilities
    """

    def __init__(self, path: str, device: Optional[str] = None) -> None:
        self.saving_path = path

        # ----------------- Load config -----------------
        args_path = os.path.join(self.saving_path, "config.yaml")
        if not os.path.isfile(args_path):
            raise FileNotFoundError(f"config.yaml not found in {self.saving_path}")
        with open(args_path, "r") as f:
            self.args: Dict[str, Any] = yaml.safe_load(f)

        self.model_args = self.args["model"]
        self.train_args = self.args["train"]
        self.data_args = self.args["data"]

        self.shape = self.data_args["shape"]  # e.g. [H, W] or [T, H, W]
        self.data_name = self.data_args["name"]

        # ----------------- Device & seed -----------------
        seed = self.train_args.get("seed", 42)
        set_seed(seed)

        if device is not None:
            self.device = torch.device(device)
        else:
            default_dev = self.train_args.get(
                "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.device = torch.device(default_dev)

        # ----------------- Build & load model -----------------
        self.model_name = self.model_args["name"]
        self.model = self.build_model()
        self.load_model()
        self.model.to(self.device)

        # Evaluator (for MSE/RMSE/PSNR/SSIM etc.)
        self.build_evaluator()

        # Normalizer / dataloaders are built on demand via build_data()
        self.normalizer = None
        self.train_loader: Optional[data.DataLoader] = None
        self.valid_loader: Optional[data.DataLoader] = None
        self.test_loader: Optional[data.DataLoader] = None

    # ------------------------------------------------------------------
    # Build components
    # ------------------------------------------------------------------
    def build_model(self, **kwargs: Any) -> torch.nn.Module:
        if self.model_name not in MODEL_REGISTRY:
            raise NotImplementedError(f"Model {self.model_name} not implemented")
        print(f"Building model: {self.model_name}")
        model_cls = MODEL_REGISTRY[self.model_name]
        return model_cls(self.model_args)

    def load_model(self, **kwargs: Any) -> None:
        model_path = os.path.join(self.saving_path, "best_model.pth")
        if os.path.isfile(model_path):
            print(f"=> loading checkpoint '{model_path}'")
            checkpoint = torch.load(model_path, map_location="cpu")
            # checkpoint may be either plain state_dict or full dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            # strict=False allows loading subset/super-set of keys
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"=> no checkpoint found at '{model_path}'")

    def build_evaluator(self) -> None:
        self.evaluator = Evaluator(self.shape)

    def build_data(self, **kwargs: Any) -> tuple[data.DataLoader, data.DataLoader, data.DataLoader, Any, Any]:
        """
        Optional: build dataset and dataloaders.

        After calling this method:
            - self.normalizer is available
            - self.train_loader / valid_loader / test_loader are set
        """
        if self.data_name not in DATASET_REGISTRY:
            raise NotImplementedError(f"Dataset {self.data_name} not implemented")

        dataset_cls = DATASET_REGISTRY[self.data_name]
        dataset = dataset_cls(self.data_args, **kwargs)

        train_loader, valid_loader, test_loader, _ = dataset.make_loaders(
            ddp=False,
            rank=0,
            world_size=1,
            drop_last=True,
        )
        
        return train_loader, valid_loader, test_loader, dataset.x_normalizer, dataset.y_normalizer

    # ------------------------------------------------------------------
    # Core inference / metrics
    # ------------------------------------------------------------------
    def inference(
        self, x: torch.Tensor, y: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        Default inference: plain forward and reshape to match y.
        Override in subclasses if needed.
        """
        return self.model(x).reshape(y.shape)

    def forecast(
        self,
        loader: data.DataLoader,
        x_normalizer: Optional[Any] = None,
        y_normalizer: Optional[Any] = None,
        return_all: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Run full-dataset forecast and compute metrics via Evaluator.
        """
        loss_record = self.evaluator.init_record()

        all_x = []
        all_y = []
        all_y_pred = []

        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                y_pred = self.inference(x, y, **kwargs)

                # Decode to physical scale
                if y_normalizer is not None and hasattr(y_normalizer, "decode"):
                    y_pred = y_normalizer.decode(y_pred)
                    y = y_normalizer.decode(y)

                if x_normalizer is not None and hasattr(x_normalizer, "decode"):
                    x = x_normalizer.decode(x[..., -1:])

                all_x.append(x.cpu())
                all_y.append(y.cpu())
                all_y_pred.append(y_pred.cpu())

        x = torch.cat(all_x, dim=0)
        y = torch.cat(all_y, dim=0)
        y_pred = torch.cat(all_y_pred, dim=0)

        self.evaluator(y_pred, y, record=loss_record)
        print(loss_record)
        
        if return_all:
            return loss_record, x, y, y_pred

        return loss_record
