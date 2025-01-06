import jax
import jax.numpy as jnp
from mlp.model import forward, init_NN_params, LayerParameters
import typer
from mlp.dataset import create_regression_dataset, create_split_dataset, Dataset
from typing import Callable
import matplotlib.pyplot as plt

app = typer.Typer()


def MSE_loss(params: list[LayerParameters], dataset: Dataset) -> jax.Array:
    return jnp.mean((forward(params, dataset.X) - dataset.y) ** 2)


loss_fn = MSE_loss


@jax.jit
def update(
    lr: float,
    params: list[LayerParameters],
    dataset: Dataset,
):
    grads = jax.grad(loss_fn)(params, dataset)

    return jax.tree_map(
        lambda p, g: p - lr * g,
        params,
        grads,
    )


@app.command()
def train(
    lr: float = 0.001,
    n_epochs: int = 10,
    layer_widths: str = typer.Option(
        "1,128,128,1",
        help="Widths of each layer in the model, comma-separated (e.g., 1,5,5,1)",
    ),
):
    """Train a neural network"""
    layer_widths: list[int] = [int(x) for x in layer_widths.split(",")]

    key = jax.random.key(42)
    data_gen_key, key = jax.random.split(key)

    raw_dataset = create_regression_dataset(key=data_gen_key, n_samples=10000)
    data_split_key, key = jax.random.split(key)

    split_dataset = create_split_dataset(key=data_split_key, raw_dataset=raw_dataset)
    train_dataset = Dataset(X=split_dataset.X_train, y=split_dataset.y_train)

    param_init_key, key = jax.random.split(key)
    params = init_NN_params(key=param_init_key, layer_widths=layer_widths)

    for _ in range(n_epochs):
        params = update(lr=lr, params=params, dataset=train_dataset)

    plt.scatter(train_dataset.X, train_dataset.y)
    plt.scatter(
        train_dataset.X, forward(params, train_dataset.X), label="Model prediction"
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    app()
