import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score
import random

def rename_italian_to_english(base_dir):
    translation = {
        "cane": "dog",
        "cavallo": "horse",
        "elefante": "elephant",
        "farfalla": "butterfly",
        "gallina": "chicken",
        "gatto": "cat",
        "mucca": "cow",
        "pecora": "sheep",
        "scoiattolo": "squirrel",
        "ragno": "spider"
    }
    for it_name, en_name in translation.items():
        it_path = os.path.join(base_dir, it_name)
        en_path = os.path.join(base_dir, en_name)
        if os.path.exists(it_path) and not os.path.exists(en_path):
            print(f"Renaming {it_name} → {en_name}")
            shutil.move(it_path, en_path)


def main():
    # =====================================================
    # Parameters
    # =====================================================
    data_dir = "/content/raw-img"
    output_dir = "/content/drive/MyDrive/saved_models"
    os.makedirs(output_dir, exist_ok=True)

    epochs = 25
    batch_size = 16
    lr = 1e-4
    early_stopping_patience = 4
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    rename_italian_to_english(data_dir)

    # =====================================================
    # Тransformation
    # =====================================================
    train_tf = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # =====================================================
    # Split data
    # =====================================================

    train_dataset = datasets.ImageFolder(data_dir, transform=train_tf)
    val_dataset   = datasets.ImageFolder(data_dir, transform=val_tf)
    num_classes   = len(train_dataset.classes)

    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    val_size = int(0.2 * len(train_dataset))
    train_indices = indices[val_size:]
    val_indices   = indices[:val_size]

    train_ds = Subset(train_dataset, train_indices)
    val_ds   = Subset(val_dataset,   val_indices)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # =====================================================
    # Model
    # =====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    # Unfreeze the last 3 blocks + classifier
    for name, param in model.named_parameters():
        if any(x in name for x in ["features.5", "features.6", "features.7", "classifier"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # =====================================================
    # Train
    # =====================================================
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ---- Validation ----
        model.eval()
        val_losses, preds, trues = [], [], []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_losses.append(loss.item())
                preds.extend(outputs.argmax(1).cpu().numpy())
                trues.extend(y.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        acc = accuracy_score(trues, preds)
        print(f"Epoch {epoch+1}: TrainLoss={avg_train_loss:.4f}, ValLoss={avg_val_loss:.4f}, ValAcc={acc:.4f}")

        scheduler.step(avg_val_loss)

        # ---- Early stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': train_dataset.classes
            }, os.path.join(output_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training finished. Best model saved to:", output_dir)


if __name__ == "__main__":
    main()
