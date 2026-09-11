import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from model import SimpleCNN
from data import get_dataloaders
from utils import set_seed

# Ordner für Ergebnisse erstellen
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

def main():
    # 1️⃣ Reproduzierbarkeit sicherstellen
    set_seed(42)

    # 2️⃣ Device auswählen (CPU oder GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 3️⃣ Daten laden
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)

    # 4️⃣ Modell, Loss-Funktion und Optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Listen für Metriken
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    epochs = 10

    # 5️⃣ Training + Validierung
    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(100 * correct / total)

        # Validierung
        model.eval()
        correct, total, val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100 * correct / total)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Acc: {train_accs[-1]:.2f}% | "
            f"Val Acc: {val_accs[-1]:.2f}%"
        )

    # 6️⃣ Modell speichern
    torch.save(model.state_dict(), "models/cnn_model.pth")

    # 7️⃣ Finale Evaluation auf dem Testdatensatz
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # Metriken speichern (Pipeline-Artefakt)
    with open("models/metrics.txt", "w") as f:
        f.write(f"Final Test Accuracy: {test_acc:.2f}%\n")

    # 8️⃣ Visualisierung
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig("plots/training_results.png")
    plt.close()

if __name__ == "__main__":
    main()
