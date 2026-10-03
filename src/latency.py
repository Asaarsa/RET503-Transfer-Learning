"""
latency.py
Template pengukuran latency/inference model.

Target pipeline pada materi perlu mempertimbangkan seluruh pipeline,
bukan hanya inference model: akuisisi, preprocess, inference,
postprocess, dan ROS2 bila digunakan.
"""

import time
import torch
from torchvision import models


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    ).to(device)
    model.eval()

    x = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        iterations = 100

        for _ in range(iterations):
            _ = model(x)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    print(f"Device: {device}")
    print(f"Average inference: {(elapsed / iterations) * 1000:.2f} ms")


if __name__ == "__main__":
    main()
