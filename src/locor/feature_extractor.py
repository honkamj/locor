"""Learned regression feature extraction model"""

from typing import Any, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp


class FeatureExtractor(eqx.Module):
    """Learned regression feature extraction model"""

    _initial_layer: eqx.nn.Linear
    _final_layer: eqx.nn.Linear
    _hidden_layers: list[eqx.nn.Linear]

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
        self._initial_layer = eqx.nn.Linear(
            n_input_channels,
            n_hidden_features[0],
            key=initial_conv_key,
        )
        self._hidden_layers = [
            eqx.nn.Linear(
                n_hidden_features[i],
                n_hidden_features[i + 1],
                key=hidden_layers_keys[i],
            )
            for i in range(len(n_hidden_features) - 1)
        ]
        self._final_layer = eqx.nn.Linear(
            n_hidden_features[-1],
            n_output_channels,
            key=final_conv_key,
        )

    def __call__(self, volume: Any) -> jnp.ndarray:
        """Forward pass through the feature extraction model"""
        features = jnp.moveaxis(volume, 1, -1)
        batch_shape = features.shape[:-1]
        features = features.reshape(-1, features.shape[-1])
        features = jax.nn.relu(jax.vmap(self._initial_layer)(features))
        for layer in self._hidden_layers:
            features = jax.nn.relu(jax.vmap(layer)(features))
        features = jax.nn.sigmoid(jax.vmap(self._final_layer)(features))
        return jnp.moveaxis(features.reshape(*batch_shape, -1), -1, 1)
