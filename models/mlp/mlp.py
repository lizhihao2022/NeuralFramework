# models/mlp/mlp.py
from typing import Any
from ..base import BaseMLP


class MLP(BaseMLP):
    def __init__(self, model_params: dict[str, Any]) -> None:
        super().__init__(
            in_dim=model_params["in_dim"],
            out_dim=model_params["out_dim"],
            hidden_dims=model_params.get("hidden_dims", None),
            activation=model_params.get("activation", "gelu"),
            dropout=model_params.get("dropout", 0.0),
            use_residual=model_params.get("use_residual", False),
            residual_proj=model_params.get("residual_proj", True),
            last_activation=model_params.get("last_activation", False),
        )
