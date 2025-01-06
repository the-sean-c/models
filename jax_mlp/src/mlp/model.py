from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random
from typing import NamedTuple


class LayerParameters(NamedTuple):
    weights: jax.Array
    biases: jax.Array


def init_NN_params(
    layer_widths: list[int], key: jax.Array = jax.random.key(42)
) -> list[LayerParameters]:
    """Generate initial parameters for an MLP with the given layer widths.

    args:
        layer_widths: List of integers specifying the number of neurons in
            each layer. Must contain at least 2 elements.
        key: JAX random key for weight initialization.

    returns:
        The weights and biases of each layer of the Neural Network.

    raises:
        ValueError: If layer_widths is empty or contains only one element.
    """
    if len(layer_widths) < 2:
        raise ValueError("layer_widths must contain at least 2 elements")

    keys = random.split(key, len(layer_widths) - 1)

    params = [
        LayerParameters(
            weights=random.normal(k, (n_in, n_out)) * jnp.sqrt(2 / n_in),
            biases=jnp.ones((n_out,)),
        )
        for k, n_in, n_out in zip(keys, layer_widths[:-1], layer_widths[1:])
    ]

    return params


def forward(params: list[LayerParameters], X: jax.Array):
    """Carry out a forward pass through the NN.

    args:
        params: Neural Network parameters.
        X: the feature array.

    returns:
        Predictions.
    """
    *hidden, last = params

    for layer in hidden:
        X = jax.nn.relu(jnp.dot(X, layer.weights) + layer.biases)

    return jnp.dot(X, last.weights) + last.biases
