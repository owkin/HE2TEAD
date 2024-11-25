import torch
from torch import nn
import numpy as np
from typing import Iterable, Optional, List
import copy as cp

import warnings

from he2tead.model.mlp import MLP, TilesMLP
from he2tead.model.extreme_layer import ExtremeLayer


class Chowder(torch.nn.Module):
    """
    Chowder module.
    Example:
        >>> module = Chowder(in_features=128, out_features=1, n_top=5, n_bottom=5)
        >>> logits, extreme_scores = module(slide, mask=mask)
        >>> scores = module.score_model(slide, mask=mask)
    Parameters
    ----------
    in_features: int
    out_features: int
        controls the number of scores and, by extension, the number of out_features
    n_top: int
    n_bottom: int
    tiles_mlp_hidden: Optional[List[int]] = None
    mlp_hidden: Optional[List[int]] = None
    mlp_dropout: Optional[List[float]] = None
    mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
    bias: bool = True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_channels: int = 1,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        super().__init__()

        if n_top is None and n_bottom is None:
            raise ValueError("At least one of `n_top` or `n_bottom` must not be None.")

        self.score_model = TilesMLP(
            in_features, hidden=tiles_mlp_hidden, bias=bias, out_features=n_channels
        )

        self.extreme_layer = ExtremeLayer(n_top=n_top, n_bottom=n_bottom)

        mlp_in_features = (n_top + n_bottom) * n_channels
        self.mlp = MLP(
            mlp_in_features,
            out_features,
            hidden=mlp_hidden,
            activation=mlp_activation,
        )
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES, 1), True for values that were padded.
        Returns
        -------
        logits, extreme_scores: Tuple[torch.Tensor, torch.Tensor]:
            (B, OUT_FEATURES), (B, N_TOP + N_BOTTOM, OUT_FEATURES)
        """
        scores = self.score_model(x=x, mask=mask)
        extreme_scores = self.extreme_layer(
            x=scores, mask=mask
        )  # (B, N_TOP + N_BOTTOM, OUT_FEATURES)

        # Apply MLP to the N_TOP + N_BOTTOM scores
        y = self.mlp(extreme_scores.reshape(-1, extreme_scores.shape[1] * extreme_scores.shape[2]))  # (B, OUT_FEATURES)

        return y, extreme_scores
