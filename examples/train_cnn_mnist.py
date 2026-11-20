# examples/train_cnn_mnist.py
"""
Convolutional Neural Network Digit Classification Benchmark with Tensile.
Demonstrates Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss, and AdamW
built from scratch with pure NumPy.
"""

from __future__ import annotations
import time
import numpy as np
import tensile as tl
from tensile import Tensor, nn, optim
from tensile.data import TensorDataset, DataLoader


def generate_synthetic_digit_data(
    num_samples: int = 400,
    img_size: int = 14,
    num_classes: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic 2D digit pattern images (1 x H x W) with deterministic class shapes."""
    rng = np.random.default_rng(seed)
    images = np.zeros((num_samples, 1, img_size, img_size), dtype=np.float64)
    labels = rng.integers(0, num_classes, size=num_samples)

    for i in range(num_samples):
        c = labels[i]
        # Base random noise
        images[i, 0] = rng.normal(0.0, 0.05, size=(img_size, img_size))

        # Geometric patterns corresponding to classes
        r = (c % 5) + 3
        # Class-specific oriented strokes
        if c % 2 == 0:
            images[i, 0, 2:img_size - 2, r] += 1.0  # Vertical line
        else:
            images[i, 0, r, 2:img_size - 2] += 1.0  # Horizontal line

        if c >= 5:
            # Add center ring
            yy, xx = np.ogrid[:img_size, :img_size]
            dist = np.sqrt((yy - img_size // 2) ** 2 + (xx - img_size // 2) ** 2)
            images[i, 0, (dist >= 2) & (dist <= 4)] += 0.8

    return images, labels


def main() -> None:
    print("=" * 68)
    print(" Tensile Pure-NumPy Convolutional Neural Network (CNN) Benchmark")
    print("=" * 68)

    # 1. Prepare Synthetic Digit Dataset (1x14x14)
    x_data, y_data = generate_synthetic_digit_data(num_samples=500, img_size=14, num_classes=10)
    split = 400
    x_train, y_train = x_data[:split], y_data[:split]
    x_val, y_val = x_data[split:], y_data[split:]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

    # 2. Build CNN Architecture
    # (1, 14, 14) -> Conv2d -> (8, 14, 14) -> MaxPool -> (8, 7, 7) -> Flatten (392) -> Linear(392, 10)
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Dropout(p=0.1),
        nn.Linear(in_features=8 * 7 * 7, out_features=10),
    )

    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.data.size for p in model.parameters())
    print(f"\nModel Architecture:\n{model}")
    print(f"Total Trainable Parameters: {total_params:,}\n")

    # 3. Training Loop
    epochs = 12
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b.data)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.data) * len(y_b.data)
            preds = np.argmax(logits.data, axis=-1)
            correct_train += int(np.sum(preds == y_b.data))
            total_train += len(y_b.data)

        train_loss /= total_train
        train_acc = 100.0 * correct_train / total_train

        # Validation Step
        model.eval()
        val_logits = model(Tensor(x_val))
        val_loss = float(criterion(val_logits, y_val).data)
        val_preds = np.argmax(val_logits.data, axis=-1)
        val_acc = 100.0 * np.sum(val_preds == y_val) / len(y_val)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.1f}%"
        )

    elapsed = time.perf_counter() - t0
    print(f"\nTraining completed in {elapsed:.2f}s ({elapsed / epochs * 1000:.1f} ms/epoch)")

    # 4. State Dict Checkpoint Test
    state = model.state_dict()
    print(f"Saved model state_dict with {len(state)} parameter tensors.")
    assert "0.weight" in state and "5.weight" in state
    print("State dict verification: PASSED!")


if __name__ == "__main__":
    main()
