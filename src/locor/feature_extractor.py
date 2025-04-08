"""Learned regression feature extraction model"""

from typing import Any, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp


class FeatureExtractor(eqx.Module):
    """Learned regression feature extraction model"""

    _initial_conv: eqx.nn.Conv
    _final_conv: eqx.nn.Conv
    _hidden_layers: list[eqx.nn.Conv]

    def __init__(
        self,
        n_dims: int,
        n_input_channels: int,
        n_hidden_features: Sequence[int],
        n_output_channels: int,
        key: jnp.ndarray,
    ) -> None:
        super().__init__()
        initial_conv_key, hidden_layers_key, final_conv_key = jax.random.split(key, 3)
        hidden_layers_keys = jax.random.split(hidden_layers_key, len(n_hidden_features) - 1)
        self._initial_conv = eqx.nn.Conv(
            num_spatial_dims=n_dims,
            in_channels=n_input_channels,
            out_channels=n_hidden_features[0],
            kernel_size=1,
            key=initial_conv_key,
        )
        self._hidden_layers = [
            eqx.nn.Conv(
                num_spatial_dims=n_dims,
                in_channels=n_hidden_features[i],
                out_channels=n_hidden_features[i + 1],
                kernel_size=1,
                key=hidden_layers_keys[i],
            )
            for i in range(len(n_hidden_features) - 1)
        ]
        self._final_conv = eqx.nn.Conv(
            num_spatial_dims=n_dims,
            in_channels=n_hidden_features[-1],
            out_channels=n_output_channels,
            kernel_size=1,
            key=final_conv_key,
        )

    def __call__(self, volume: Any) -> jnp.ndarray:
        """Forward pass through the feature extraction model"""
        features = jax.nn.relu(jax.vmap(self._initial_conv)(volume))
        for layer in self._hidden_layers:
            features = jax.nn.relu(jax.vmap(layer)(features))
        return jax.nn.sigmoid(jax.vmap(self._final_conv)(features))
