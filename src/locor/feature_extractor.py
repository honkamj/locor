"""Learned regression feature extraction model"""

from typing import Sequence

import torch.nn
from torch import Tensor
from torch.nn import Module, ModuleList


class FeatureExtractor(Module):
    """Learned regression feature extraction model"""

    def __init__(
        self,
        n_dims: int,
        n_input_channels: int,
        n_hidden_features: Sequence[int],
        n_output_channels: int,
    ) -> None:
        super().__init__()
        self._n_input_channels = n_input_channels
        self._n_output_channels = n_output_channels
        self._initial_conv = self._conv_nd(n_dims)(
            n_input_channels, n_hidden_features[0], kernel_size=1
        )
        self._hidden_layers = ModuleList(
            [
                self._conv_nd(n_dims)(n_hidden_features[i], n_hidden_features[i + 1], kernel_size=1)
                for i in range(len(n_hidden_features) - 1)
            ]
        )
        self._final_conv = self._conv_nd(n_dims)(
            n_hidden_features[-1], n_output_channels, kernel_size=1
        )

    def forward(self, volume: Tensor) -> Tensor:
        """Forward pass through the feature extraction model"""
        features = self._initial_conv(volume).relu()
        for layer in self._hidden_layers:
            features = layer(features).relu()
        return self._final_conv(features).sigmoid()

    @staticmethod
    def _conv_nd(n_dims: int):
        if not isinstance(n_dims, int):
            raise ValueError("n_dims must be an integer")
        return getattr(torch.nn, f"Conv{n_dims}d")
