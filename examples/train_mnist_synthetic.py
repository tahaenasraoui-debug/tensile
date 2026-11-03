"""
End-to-end executable demonstration of Tensile on a synthetic 10-class
MNIST-dimensional dataset (1000 samples, 784 features, 10 classes).
"""

from __future__ import annotations
import numpy as np
import tensile as tl
from tensile import nn
from tensile.data import TensorDataset, DataLoader


def generate_synthetic_mnist(
    num_samples: int = 1000,
    num_features: int = 784,
    num_classes: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesizes a learnable classification dataset with distinct class prototypes
    in 784-dimensional space plus Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.normal(loc=0.0, scale=1.5, size=(num_classes, num_features))
    labels = rng.integers(low=0, high=num_classes, size=(num_samples,), dtype=np.int64)
    noise = rng.normal(loc=0.0, scale=2.0, size=(num_samples, num_features))
    features = centroids[labels] + noise
    return features.astype(np.float64), labels


def main() -> None:
    np.random.seed(42)

    # 1. Prepare synthetic dataset and DataLoader
    X, y = generate_synthetic_mnist(num_samples=1000, num_features=784, num_classes=10)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Instantiate MLP model matching Target API
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    # 3. Instantiate Adam optimizer
    optimizer = tl.optim.Adam(model.parameters(), lr=1e-3)

    # 4. Training loop
    epochs = 15
    print("Starting Tensile Training on Synthetic MNIST (1000x784 -> 10 classes)...")
    print("-" * 68)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in dataloader:
            logits = model(x_batch)
            loss = nn.cross_entropy(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = x_batch.shape[0]
            epoch_loss += float(loss.data.item()) * batch_size
            preds = np.argmax(logits.data, axis=-1)
            correct += int(np.sum(preds == y_batch.data))
            total += batch_size

        avg_loss = epoch_loss / total
        accuracy = (correct / total) * 100.0
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:6.2f}%")

    print("-" * 68)
    print("Training complete.")


if __name__ == "__main__":
    main()
