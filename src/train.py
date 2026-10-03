"""
train.py
Eksperimen transfer learning RET503 menggunakan ResNet-18.

Mode:
  feature : backbone frozen, hanya FC dilatih
  partial : layer4 + FC dilatih
  scratch : seluruh model dilatih dari bobot acak

Contoh:
  python src/train.py --mode feature
  python src/train.py --mode partial
  python src/train.py --mode scratch
"""

import argparse
import torch
from torch import nn
from torchvision import models


def build_model(mode: str, num_classes: int):
    if mode not in {"feature", "partial", "scratch"}:
        raise ValueError("mode harus feature, partial, atau scratch")

    if mode == "scratch":
        model = models.resnet18(weights=None)
        for p in model.parameters():
            p.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    if mode == "feature":
        for p in model.parameters():
            p.requires_grad = False

    elif mode == "partial":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["feature", "partial", "scratch"],
        required=True,
    )
    parser.add_argument("--num-classes", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.mode, args.num_classes).to(device)

    if args.mode == "feature":
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
        )
    elif args.mode == "partial":
        optimizer = torch.optim.Adam(
            [
                {"params": model.layer4.parameters(), "lr": 1e-4},
                {"params": model.fc.parameters(), "lr": 1e-3},
            ]
        )
    else:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
        )

    print(f"Mode     : {args.mode}")
    print(f"Device   : {device}")
    print(f"Optimizer: {optimizer}")
    print("TODO: tambahkan DataLoader, augmentation, training loop, validation,")
    print("      checkpoint, scheduler, dan pencatatan hasil.")


if __name__ == "__main__":
    main()
