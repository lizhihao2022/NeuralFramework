# trainers/base.py
import os
import logging
from functools import partial
from typing import Any, Dict, Optional

import inspect
import torch
import torch.distributed as dist
import wandb

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

from utils import LossRecord, LpLoss, CompositeLoss, build_loss_fn, Evaluator, autoregressive_rollout

from models import MODEL_REGISTRY
from datasets import DATASET_REGISTRY


class BaseTrainer:
    """
    Base trainer with single-GPU / DDP support and common training loop.
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        self.args = args
        self.model_args = args["model"]
        self.data_args = args["data"]
        self.optim_args = args["optimize"]
        self.scheduler_args = args["schedule"]
        self.train_args = args["train"]
        self.log_args = args["log"]

        # ------------------------------------------------------------------
        # Distributed setup
        # ------------------------------------------------------------------
        self._setup_distribute()

        # Optional geometry / coordinates from dataset
        self.geom: Optional[Dict[str, Any]] = None
        self.coords: Optional[torch.Tensor] = None

        # Flags describing which extra arguments model.forward accepts
        self._model_accepts_coords: bool = False
        self._model_accepts_geom: bool = False
        self._model_accepts_y: bool = False

        # Logger & wandb
        self.logger = logging.info if self.log_args.get("log", True) else print
        self.wandb = self.log_args.get("wandb", False)

        if self._is_main_process() and self.wandb:
            wandb.init(
                project=self.log_args.get("wandb_project", "default"),
                name=self.train_args.get("saving_name", "experiment"),
                tags=[
                    self.model_args.get("name", "model"),
                    self.data_args.get("name", "dataset"),
                ],
                config=args,
            )

        # ------------------------------------------------------------------
        # Build model / optimizer / scheduler
        # ------------------------------------------------------------------
        self.model_name = self.model_args["name"]
        self.main_log(f"Building {self.model_name} model")
        self.model = self.build_model()
        self.apply_init()

        self.model = self.model.to(self.device)

        # Wrap with DP / DDP if needed
        if self.dp:
            self.device_ids = self.train_args.get(
                "device_ids", list(range(torch.cuda.device_count()))
            )
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
            self.main_log(f"Using DataParallel with GPUs: {self.device_ids}")
        elif self.ddp:
            # local_rank/device is set in _setup_distribute
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        # Ensure all params are contiguous
        for p in self.model.parameters():
            if not p.is_contiguous():
                p.data = p.data.contiguous()

        # Inspect model.forward signature once
        self._inspect_model_signature()

        self.start_epoch = 0
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # Optionally resume from checkpoint
        if self.train_args.get("load_ckpt", False):
            self.load_ckpt(self.train_args["ckpt_path"])

        self.loss_fn = self.build_loss()
        self.evaluator = self.build_evaluator()

        self.main_log(f"Model: {self.model}")
        self.main_log(
            "Model parameters: {:.2f}M".format(
                sum(p.numel() for p in self._unwrap().parameters()) / 1e6
            )
        )
        self.main_log(f"Optimizer: {self.optimizer}")
        self.main_log(f"Scheduler: {self.scheduler}")

        # ------------------------------------------------------------------
        # Data
        # ------------------------------------------------------------------
        self.data = self.data_args["name"]
        self.main_log(f"Loading {self.data} dataset")
        self.build_data()
        self.main_log(f"Train dataset size: {self.train_length}")
        self.main_log(f"Valid dataset size: {self.valid_length}")
        self.main_log(f"Test  dataset size: {self.test_length}")

        # ------------------------------------------------------------------
        # Training hyperparameters
        # ------------------------------------------------------------------
        self.epochs = self.train_args["epochs"]
        self.eval_freq = self.train_args["eval_freq"]
        self.patience = self.train_args["patience"]

        self.saving_best = self.train_args.get("saving_best", True)
        self.saving_ckpt = self.train_args.get("saving_ckpt", False)
        self.ckpt_freq = self.train_args.get("ckpt_freq", 100)
        self.ckpt_max = self.train_args.get("ckpt_max", 5)
        self.saving_path = self.train_args.get("saving_path", None)

    # ----------------------------------------------------------------------
    # Distributed helpers
    # ----------------------------------------------------------------------
    def _setup_distribute(self) -> None:
        """
        Decide whether we are in DDP / DP / single-GPU mode and set self.device.
        """
        self.world_size = self.train_args.get("world_size", 1)
        self.dist_mode = self.train_args.get("distribute_mode", "DDP")

        self.ddp = (
            self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
            and self.dist_mode == "DDP"
        )
        # DataParallel: single process, multiple GPUs
        self.dp = (
            self.dist_mode == "DP"
            and not self.ddp
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
        )
        self.dist = self.ddp or self.dp

        if self.ddp:
            self.local_rank = self.train_args.get("local_rank", 0)
            self.device = torch.device(
                f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.local_rank = self.train_args.get("device_ids", [0])[0]
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _unwrap(self) -> nn.Module:
        """
        Return the underlying model (strip DDP / DataParallel).
        """
        if isinstance(self.model, (DDP, nn.DataParallel)):
            return self.model.module
        return self.model

    def _is_main_process(self) -> bool:
        """
        True only for rank 0 in DDP; always True for DP / single-GPU.
        """
        if not self.dist:
            return True
        if self.dp:  # single process, multiple GPUs
            return True
        # DDP: only rank 0
        return self.local_rank == 0

    # ----------------------------------------------------------------------
    # Model signature inspection / unified forward
    # ----------------------------------------------------------------------
    def _inspect_model_signature(self) -> None:
        """
        Inspect model.forward once and record which extra arguments
        are supported (coords / geom / y).
        """
        base_model = self._unwrap()
        try:
            sig = inspect.signature(base_model.forward)
        except (TypeError, ValueError):
            # Fallback: assume no extra args
            self._model_accepts_coords = False
            self._model_accepts_geom = False
            self._model_accepts_y = False
            return

        params = sig.parameters
        self._model_accepts_coords = "coords" in params
        self._model_accepts_geom = "geom" in params
        self._model_accepts_y = "y" in params

        self.main_log(
            f"Model forward supports: "
            f"coords={self._model_accepts_coords}, "
            f"geom={self._model_accepts_geom}, "
            f"y={self._model_accepts_y}"
        )

    def _forward_model(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **extra_kwargs: Any,
    ) -> torch.Tensor:
        """
        Unified forwarding entry for training / evaluation.

        It automatically injects:
          - coords: batched coordinates if model.forward(coords=...) exists.
          - geom: geometry dict if model.forward(geom=...) exists.
          - y: target tensor if model.forward(y=...) exists.

        Args:
            x:  input tensor (e.g. B x N x C)
            y:  optional target tensor
            extra_kwargs: extra keyword args passed through to model.forward

        Returns:
            Model prediction tensor.
        """
        kwargs: Dict[str, Any] = dict(extra_kwargs)

        # Inject coordinates (if available and requested by the model)
        if self._model_accepts_coords and self.coords is not None:
            B = x.size(0)
            coords = self.coords
            # coords is typically (N_points, d); broadcast to (B, N_points, d)
            if coords.dim() == 2:
                coords_batched = coords.unsqueeze(0).expand(B, -1, -1)
            else:
                coords_batched = coords
            kwargs["coords"] = coords_batched

        # Inject geometry (if requested)
        if self._model_accepts_geom and self.geom is not None:
            kwargs["geom"] = self.geom

        # Inject y if the model wants it (e.g. diffusion-style models)
        if self._model_accepts_y and (y is not None):
            kwargs["y"] = y

        return self.model(x, **kwargs)

    # ----------------------------------------------------------------------
    # Init / build components
    # ----------------------------------------------------------------------
    def get_initializer(self, name: Optional[str]):
        if name is None:
            return None
        if name == "xavier_normal":
            return partial(torch.nn.init.xavier_normal_)
        if name == "kaiming_uniform":
            return partial(torch.nn.init.kaiming_uniform_)
        if name == "kaiming_normal":
            return partial(torch.nn.init.kaiming_normal_)
        raise ValueError(f"Unknown initializer: {name}")

    def apply_init(self) -> None:
        initializer = self.get_initializer(self.train_args.get("initializer", None))
        if initializer is None:
            return

        def init_module(module: nn.Module) -> None:
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.dim() > 1:
                with torch.no_grad():
                    initializer(weight)

        self._unwrap().apply(init_module)
        self.main_log(f"Apply {self.train_args.get('initializer')} initializer")

    def build_model(self, **kwargs: Any) -> nn.Module:
        if self.model_args["name"] not in MODEL_REGISTRY:
            raise NotImplementedError(
                f"Model {self.model_args['name']} not implemented"
            )
        model_cls = MODEL_REGISTRY[self.model_args["name"]]
        return model_cls(self.model_args)

    def build_optimizer(self, **kwargs: Any):
        opt_name = self.optim_args["optimizer"]
        if opt_name == "Adam":
            return torch.optim.Adam(
                self._unwrap().parameters(),
                lr=self.optim_args["lr"],
                weight_decay=self.optim_args["weight_decay"],
            )
        if opt_name == "SGD":
            return torch.optim.SGD(
                self._unwrap().parameters(),
                lr=self.optim_args["lr"],
                momentum=self.optim_args.get("momentum", 0.9),
                weight_decay=self.optim_args["weight_decay"],
            )
        if opt_name == "AdamW":
            return torch.optim.AdamW(
                self._unwrap().parameters(),
                lr=self.optim_args["lr"],
                weight_decay=self.optim_args["weight_decay"],
            )
        raise NotImplementedError(f"Optimizer {opt_name} not implemented")

    def build_scheduler(self, **kwargs: Any):
        sch_name = self.scheduler_args.get("scheduler", None)
        if sch_name is None:
            return None

        if sch_name == "MultiStepLR":
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.scheduler_args["milestones"],
                gamma=self.scheduler_args["gamma"],
            )
        if sch_name == "OneCycleLR":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.optim_args["lr"],
                div_factor=self.scheduler_args["div_factor"],
                final_div_factor=self.scheduler_args["final_div_factor"],
                pct_start=self.scheduler_args["pct_start"],
                steps_per_epoch=self.scheduler_args["steps_per_epoch"],
                epochs=self.train_args["epochs"],
            )
        if sch_name == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_args["step_size"],
                gamma=self.scheduler_args["gamma"],
            )

        raise NotImplementedError(f"Scheduler {sch_name} not implemented")

    def build_loss(self, **kwargs: Any):
        # Relative Lp loss, averaged over batch
        loss_cfg = self.args.get("loss", None)
        if loss_cfg is None:
            return LpLoss(size_average=True)
        return build_loss_fn(loss_cfg)

    def build_evaluator(self):
        # Configurable evaluation metrics (see YAML: `evaluate` section).
        eval_args = (self.args.get("evaluate") or {})
        metric_cfg = eval_args.get("metrics", None)
        strict = bool(eval_args.get("strict", True))
        rollout_args = (eval_args.get("rollout") or {})
        rollout_per_step = bool(rollout_args.get("per_step", True))

        # Optional global kwargs for metrics
        metric_kwargs = eval_args.get("metric_kwargs", None) or eval_args.get("kwargs", None) or {}

        return Evaluator(
            shape=self.data_args.get("shape"),
            metric_cfg=metric_cfg,
            strict=strict,
            rollout_per_step=rollout_per_step,
            **metric_kwargs,
        )

    def build_data(self, **kwargs: Any) -> None:
        if self.data_args["name"] not in DATASET_REGISTRY:
            raise NotImplementedError(
                f"Dataset {self.data_args['name']} not implemented"
            )
        dataset_cls = DATASET_REGISTRY[self.data_args["name"]]
        dataset = dataset_cls(self.data_args)

        # Geometry / coordinates (if dataset provides them)
        self.geom = getattr(dataset, "geom", None)
        self.coords = getattr(dataset, "coords", None)
        if self.coords is not None:
            # Move coordinates to current device once
            self.coords = self.coords.to(self.device)
        else:
            self.main_log("Dataset does not provide coordinates tensor")

        if self.geom is not None:
            self.main_log(f"Dataset geometry: {self.geom}")
        else:
            self.main_log("Dataset does not provide geometry dict")

        self.train_loader, self.valid_loader, self.test_loader, self.train_sampler = dataset.make_loaders(
            ddp=self.ddp,
            rank=self.local_rank,
            world_size=self.world_size,
            drop_last=True,
        )

        self.train_length = len(dataset.train_dataset)
        self.valid_length = len(dataset.valid_dataset)
        self.test_length = len(dataset.test_dataset)

        self.x_normalizer = dataset.x_normalizer
        self.y_normalizer = dataset.y_normalizer

    # ----------------------------------------------------------------------
    # Checkpoint IO
    # ----------------------------------------------------------------------
    def _get_state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        model_to_save = self._unwrap()
        return {k: v.detach().cpu() for k, v in model_to_save.state_dict().items()}

    def save_ckpt(self, epoch: int) -> None:
        if self.saving_path is None:
            return
        os.makedirs(self.saving_path, exist_ok=True)
        state_dict_cpu = self._get_state_dict_cpu()

        state = {
            "epoch": epoch,
            "model_state_dict": state_dict_cpu,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        ckpt_path = os.path.join(self.saving_path, f"model_epoch_{epoch}.pth")
        torch.save(state, ckpt_path)

        # Keep only the last N checkpoints
        if self.ckpt_max is not None and self.ckpt_max > 0:
            ckpt_list = [
                f
                for f in os.listdir(self.saving_path)
                if f.startswith("model_epoch_") and f.endswith(".pth")
            ]
            if len(ckpt_list) > self.ckpt_max:
                ckpt_list.sort(
                    key=lambda x: int(x.split("_")[-1].split(".")[0])
                )
                os.remove(os.path.join(self.saving_path, ckpt_list[0]))

    def save_model(self, model_path: str) -> None:
        state_dict_cpu = self._get_state_dict_cpu()
        torch.save(state_dict_cpu, model_path)
        self.main_log(f"Save model to {model_path}")

    def load_model(self, model_path: str) -> None:
        state = torch.load(model_path, map_location="cpu")
        self._unwrap().load_state_dict(state)
        self.main_log(f"Load model from {model_path}")

    def load_ckpt(self, ckpt_path: str) -> None:
        """
        Load checkpoint (model + optimizer + scheduler + epoch).
        """
        state = torch.load(ckpt_path, map_location="cpu")

        model_state = state.get("model_state_dict", None)
        if model_state is not None:
            self._unwrap().load_state_dict(model_state)

        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            # Move optimizer state tensors to correct device
            for group in self.optimizer.state.values():
                for k, v in group.items():
                    if isinstance(v, torch.Tensor):
                        group[k] = v.to(self.device)

        if "scheduler_state_dict" in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        self.start_epoch = state.get("epoch", 0) + 1
        self.main_log(
            f"Load checkpoint from {ckpt_path}, epoch {state.get('epoch', 'N/A')}"
        )

    # ----------------------------------------------------------------------
    # Logging helpers
    # ----------------------------------------------------------------------
    def main_log(self, msg: str) -> None:
        if self._is_main_process():
            self.logger(msg)

    # ----------------------------------------------------------------------
    # Core training loop
    # ----------------------------------------------------------------------
    def process(self, **kwargs: Any) -> None:
        self.main_log("Start training")
        best_epoch = 0
        best_metrics: Optional[Dict[str, float]] = None
        best_path = (
            os.path.join(self.saving_path, "best_model.pth")
            if self.saving_path is not None
            else "best_model.pth"
        )
        counter = 0

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        bar = (
            tqdm(total=self.epochs - self.start_epoch)
            if self._is_main_process()
            else None
        )

        for epoch in range(self.start_epoch, self.epochs):
            train_loss_record = self.train(epoch)
            self.main_log(
                "Epoch {} | {} | lr: {:.4e}".format(
                    epoch,
                    train_loss_record,
                    self.optimizer.param_groups[0]["lr"],
                )
            )
            if self._is_main_process() and self.wandb:
                wandb.log(
                    {
                        **train_loss_record.to_dict(),
                        "epoch": epoch,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            if self._is_main_process() and self.saving_ckpt and (epoch + 1) % self.ckpt_freq == 0:
                self.save_ckpt(epoch)
                self.main_log(f"Epoch {epoch} | save checkpoint in {self.saving_path}")

            if (epoch + 1) % self.eval_freq == 0:
                valid_loss_record = self.evaluate(split="valid")
                self.main_log(f"Epoch {epoch} | {valid_loss_record}")
                valid_metrics = valid_loss_record.to_dict()

                if self._is_main_process() and self.wandb:
                    wandb.log(
                        {**valid_metrics, "epoch": epoch, "phase": "valid"}
                    )

                if (not best_metrics) or (
                    valid_metrics["valid_loss"] < best_metrics["valid_loss"]
                ):
                    counter = 0
                    best_epoch = epoch
                    best_metrics = valid_metrics
                    if self._is_main_process() and self.saving_best:
                        self.save_model(best_path)
                elif self.patience != -1:
                    counter += 1
                    if counter >= self.patience:
                        self.main_log(f"Early stop at epoch {epoch}")
                        # Early stop across all ranks
                        stop_flag = torch.tensor(
                            [1 if self._is_main_process() else 0],
                            device=self.device,
                            dtype=torch.int32,
                        )
                        if self.ddp and dist.is_initialized():
                            dist.broadcast(stop_flag, src=0)
                        if stop_flag.item() > 0:
                            break

            if self._is_main_process() and bar is not None:
                bar.update(1)

        if self._is_main_process() and bar is not None:
            bar.close()
        self.main_log("Optimization Finished!")

        # No validation ever run -> save final model
        if self._is_main_process() and not best_metrics:
            self.save_model(best_path)

        if self.ddp and dist.is_initialized():
            dist.barrier()

        # Load best model for final evaluation (all ranks)
        self.load_model(best_path)

        valid_loss_record = self.evaluate(split="valid")
        self.main_log(f"Valid metrics: {valid_loss_record}")
        test_loss_record = self.evaluate(split="test")
        self.main_log(f"Test metrics: {test_loss_record}")

        if self._is_main_process() and self.wandb:
            run = wandb.run
            if run is not None:
                run.summary["best_epoch"] = best_epoch
                run.summary.update(test_loss_record.to_dict())
            wandb.finish()

        if self.ddp and dist.is_initialized():
            dist.barrier()

    # ----------------------------------------------------------------------
    # Train / eval
    # ----------------------------------------------------------------------
    def train(self, epoch: int, **kwargs: Any) -> LossRecord:
        loss_record = LossRecord(["train_loss"])

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        self.model.train()
        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Unified forward (auto-inject coords / geom / y)
            y_pred = self._forward_model(x, y)

            loss_kwargs = {"x": x, "coords": self.coords, "geom": self.geom}
            if isinstance(self.loss_fn, CompositeLoss):
                loss, logs = self.loss_fn(y_pred, y, return_dict=True)
                update_dict = {"train_loss": float(loss.item())}
                for k, v in logs.items(): # type: ignore
                    update_dict[f"train_{k}"] = float(v)
                loss_record.update(update_dict, n=x.size(0))
            else:
                loss = self.loss_fn(y_pred, y, **loss_kwargs)
                loss_record.update({"train_loss": float(loss.item())}, n=x.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # Global average over DDP
        if self.ddp and dist.is_initialized():
            loss_record.dist_reduce(device=self.device)

        return loss_record

    def _one_step(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        """
        One-step inference: unified forward.
        Override in subclasses if needed.
        """
        y_pred = self._forward_model(x, y, **kwargs)
        return y_pred
    
    def inference(self, x: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Default inference: unified forward, reshaped to match y.
        Override in subclasses if needed.
        x: (B, N, C_in)
        y: one-step -> (B, N, C_out)
           rollout -> (B, S, N, C_out)
        """
        if y.ndim == 3: # one_step
            y_pred = self._one_step(x, y, **kwargs)
            return y_pred.reshape_as(y)
        elif y.ndim == 4: # autoregressive rollout
            b, s, n, c = y.shape
            u0 = x[..., -1:].contiguous() # (B, N, C_in)
            seq = autoregressive_rollout(self._one_step, u0, y, steps=s)
            return seq.reshape_as(y)
        
        raise ValueError(f"Unsupported y shape for inference: {tuple(y.shape)}")

    def evaluate(self, split: str = "valid", **kwargs: Any) -> LossRecord:
        if split == "valid":
            eval_loader = self.valid_loader
        elif split == "test":
            eval_loader = self.test_loader
        else:
            raise ValueError("split must be 'valid' or 'test'")

        loss_record = self.evaluator.init_record([f"{split}_loss"])

        all_y = []
        all_y_pred = []

        self.model.eval()
        with torch.no_grad():
            for x, y in eval_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                y_pred = self.inference(x, y, **kwargs)

                all_y.append(y)
                all_y_pred.append(y_pred)

        y = torch.cat(all_y, dim=0)
        y_pred = torch.cat(all_y_pred, dim=0)
        
        loss_kwargs = {"x": None, "coords": self.coords, "geom": self.geom}
        if isinstance(self.loss_fn, CompositeLoss):
            loss, logs = self.loss_fn(y_pred, y, return_dict=True, **loss_kwargs)
        else:
            loss = self.loss_fn(y_pred, y, **loss_kwargs)
            logs = None

        total_samples = y.size(0)
        loss_record.update({f"{split}_loss": float(loss.item())}, n=total_samples)
        if logs is not None:
            loss_record.update({f"{split}_{k}": float(v) for k, v in logs.items()}, n=total_samples) # type: ignore

        if self.y_normalizer is not None and hasattr(self.y_normalizer, "decode"):
            y_pred = self.y_normalizer.decode(y_pred)
            y = self.y_normalizer.decode(y)
        self.evaluator(y_pred, y, record=loss_record, batch_size=total_samples)

        if self.ddp and dist.is_initialized():
            loss_record.dist_reduce(device=self.device)

        return loss_record
