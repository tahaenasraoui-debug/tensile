"""
Dataset and DataLoader abstractions for batching and shuffling data.
"""

from __future__ import annotations
from typing import Any, Iterator, Union
import numpy as np
from tensile.tensor import Tensor


class Dataset:
    """Abstract base class for all datasets."""

    def __len__(self) -> int:
        raise NotImplementedError("Dataset subclasses must implement __len__().")

    def __getitem__(self, index: Union[int, np.ndarray, slice]) -> Any:
        raise NotImplementedError("Dataset subclasses must implement __getitem__().")


class TensorDataset(Dataset):
    """
    Dataset wrapping NumPy arrays or Tensors into indexable tuples (e.g., (x, y)).
    All arrays must share the same first dimension size (number of samples).
    """

    def __init__(self, *tensors: Union[np.ndarray, Tensor]) -> None:
        if len(tensors) == 0:
            raise ValueError("TensorDataset requires at least one array or Tensor.")

        arrays = [t.data if isinstance(t, Tensor) else np.asarray(t) for t in tensors]
        first_len = arrays[0].shape[0]
        if not all(arr.shape[0] == first_len for arr in arrays):
            raise ValueError("All tensors must have the same size along dimension 0.")

        self.tensors: tuple[np.ndarray, ...] = tuple(arrays)

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    def __getitem__(self, index: Union[int, np.ndarray, slice]) -> tuple[np.ndarray, ...]:
        return tuple(arr[index] for arr in self.tensors)


class DataLoader:
    """
    Iterable batch generator over a Dataset, yielding batches wrapped as
    `Tensor(..., requires_grad=False)`.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Union[Tensor, tuple[Tensor, ...]]]:
        n_samples = len(self.dataset)
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n_samples, self.batch_size):
            batch_indices = indices[start_idx : start_idx + self.batch_size]

            if isinstance(self.dataset, TensorDataset):
                batch_items = self.dataset[batch_indices]
                yield tuple(Tensor(arr, requires_grad=False) for arr in batch_items)
            else:
                samples = [self.dataset[int(i)] for i in batch_indices]
                if isinstance(samples[0], tuple):
                    num_fields = len(samples[0])
                    yield tuple(
                        Tensor(np.stack([s[f] for s in samples], axis=0), requires_grad=False)
                        for f in range(num_fields)
                    )
                else:
                    yield Tensor(np.stack(samples, axis=0), requires_grad=False)
