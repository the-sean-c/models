import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generator

import jax
import jax.numpy as jnp

key = jax.random.key(42)


@dataclass
class Dataset:
    X: jax.Array
    y: jax.Array

    def __post_init__(self):
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have same length")

    def __len__(self):
        return len(self.X)


@dataclass
class SplitDataset:
    X_train: jax.Array
    y_train: jax.Array
    X_test: jax.Array
    y_test: jax.Array
    X_valid: jax.Array
    y_valid: jax.Array

    def __post_init__(self):
        is_same_length = (
            len(self.X_train) == len(self.y_train)
            and len(self.X_test) == len(self.y_test)
            and len(self.X_valid) == len(self.y_valid)
        )
        is_same_features = (
            self.X_train.shape[1] == self.X_test.shape[1] == self.X_valid.shape[1]
            and self.y_train.shape[1] == self.y_test.shape[1] == self.y_valid.shape[1]
        )

        if not is_same_length:
            raise ValueError("X and y must have the same length")

        if not is_same_features:
            raise ValueError(
                "X's and y's must respectively have same number of features."
            )


def create_regression_dataset(
    key: jax.Array, n_samples: int = 1000, noise: float = 0.3
) -> Dataset:
    """Create a Regression Dataset.

    Dataset has a single feature, and f(X) is a sine wave with normally
    distributed noise.

    Args:
        key: A pseudorandom number generator key.
        n_features: The number of regression features.
        n_samples: The number of samples.
        noise: Standard deviation of normally distributed noise.

    Returns:
        A dataset containing X and y, the independent and dependent variable
        arrays.
    """
    key, subkey = jax.random.split(key)
    X = jax.random.normal(subkey, (n_samples,)) * 5
    key, subkey = jax.random.split(key)
    y = jnp.sin(X) + jax.random.normal(subkey, (n_samples,)) * noise

    return Dataset(X=X, y=y)


def split_dataset(
    key: jax.Array,
    raw_dataset: Dataset,
    test_portion: float = 0.2,
    valid_portion: float = 0.2,
) -> SplitDataset:
    """Split Dataset into train, test, valid splits.

    Args:
        key: a pseudorandom number generator key.
        raw_dataset: raw independent and dependent variable arrays.
        test_portion: portion assigned to test set in [0, 0.4]
        valid_portion: portion assigned to validation set in [0, 0.4]

    Returns:
        A dataset with train, test and valid splits.
    """
    if (
        test_portion < 0
        or test_portion > 0.4
        or valid_portion < 0
        or valid_portion > 0.4
    ):
        raise ValueError("portions must be [0, 0.4]")

    n_samples = len(raw_dataset)
    n_test = n_samples * test_portion // 1
    n_valid = n_samples * valid_portion // 1
    n_train = n_samples - n_test - n_valid

    key, subkey = jax.random.split(key)
    shuffled_idx = jax.random.permutation(subkey, n_samples)

    train_idx = shuffled_idx[:n_train]
    test_idx = shuffled_idx[n_train : n_train + n_test]
    valid_idx = shuffled_idx[n_train + n_test :]

    return SplitDataset(
        X_train=raw_dataset.X[train_idx],
        y_train=raw_dataset.y[train_idx],
        X_test=raw_dataset.X[test_idx],
        y_test=raw_dataset.y[test_idx],
        X_valid=raw_dataset.X[valid_idx],
        y_valid=raw_dataset.y[valid_idx],
    )


def generate_data_batches(
    key: jax.Array, dataset: Dataset, batch_size: int, n_batches: int = 1
) -> Generator[Dataset, None, None]:
    """Generate batches of data.

    Args:
        key: a pseudorandom number generator key.
        X_train: independent feature array.
        y_train: dependent feature array.
        batch_size: number of observations in a batch.

    Yields:
        Batched datasets
    """
    shuffled_idxs = jax.random.permutation(key, len(dataset))

    for i in range(0, (len(dataset) // (batch_size * n_batches)) + 1):
        batches = []
        for j in range(n_batches):
            start_idx = i * batch_size * n_batches + j * batch_size
            idxs = shuffled_idxs[start_idx : start_idx + batch_size]
            batches.append(Dataset(X=dataset.X[idxs], y=dataset.y[idxs]))

        yield Dataset(X=dataset.X[idxs], y=dataset.y[idxs])
