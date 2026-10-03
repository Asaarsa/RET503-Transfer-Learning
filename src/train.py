"""Training ResNet-18 untuk klasifikasi baut vs mur."""
import argparse
import csv
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

CLASSES = ["baut", "mur"]
DATASET_DIR = Path('dataset')
RESULTS_DIR = Path('results')
MODELS_DIR = Path('models')

def build_model(mode, num_classes):
    if mode == 'scratch':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    if mode == 'partial':
        for p in model.layer4.parameters():
            p.requires_grad = True
    elif mode != 'feature':
        raise ValueError('mode harus feature, partial, atau scratch')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def make_loaders(batch_size):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = datasets.ImageFolder(DATASET_DIR / 'train', transform=train_tf)
    val_ds = datasets.ImageFolder(DATASET_DIR / 'val', transform=val_tf)
    if train_ds.classes != CLASSES or val_ds.classes != CLASSES:
        raise RuntimeError(f'Kelas harus {CLASSES}. Train={train_ds.classes}, Val={val_ds.classes}')
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2))

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train); total_loss = 0.0; correct = 0; total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            outputs = model(images); loss = criterion(outputs, labels)
            if train: loss.backward(); optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item(); total += labels.size(0)
    return total_loss / total, correct / total

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['feature','partial','scratch'], required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = make_loaders(args.batch_size)
    model = build_model(args.mode, len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    if args.mode == 'partial':
        optimizer = torch.optim.Adam([
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3}])
    else:
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    RESULTS_DIR.mkdir(exist_ok=True); (MODELS_DIR / args.mode).mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / 'accuracy.csv'
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    start = time.perf_counter(); best_val = 0.0; best_epoch = 0
    with csv_path.open('a', newline='') as f:
        writer = csv.writer(f)
        if write_header: writer.writerow(['mode','epoch','train_accuracy','val_accuracy'])
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
            scheduler.step()
            writer.writerow([args.mode, epoch, f'{train_acc:.6f}', f'{val_acc:.6f}']); f.flush()
            print(f'[{args.mode}] epoch {epoch}/{args.epochs} train_acc={train_acc:.4f} val_acc={val_acc:.4f}')
            if val_acc > best_val:
                best_val, best_epoch = val_acc, epoch
                torch.save({'mode':args.mode,'classes':CLASSES,'model_state_dict':model.state_dict(),'val_accuracy':best_val,'epoch':best_epoch}, MODELS_DIR / args.mode / 'best.pt')
    elapsed = time.perf_counter() - start
    path = RESULTS_DIR / 'training_time.csv'; header = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='') as f:
        w = csv.writer(f)
        if header: w.writerow(['mode','training_time_seconds','best_val_accuracy','best_epoch'])
        w.writerow([args.mode, f'{elapsed:.3f}', f'{best_val:.6f}', best_epoch])
    print(f'Training selesai: {elapsed:.2f} s; best val accuracy={best_val:.4f} epoch={best_epoch}')

if __name__ == '__main__': main()