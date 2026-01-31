import torch
import torch.nn as nn

import os
import csv

from model import NN
from data import get_loaders
from optimizers import get_optimizer
from config import EPOCHS, BATCH_SIZE, EXPS, SEED


os.makedirs("results", exist_ok=True)


def logger(name, epoch, train_loss, test_loss, test_acc, lr):
    filepath = f"results/{name}.csv"
    filexists = os.path.isfile(filepath)

    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not filexists:
            writer.writerow(["epoch", "train_loss", "test_loss", "test_acc", "lr"])

        writer.writerow([
            epoch,
            train_loss,
            test_loss,
            test_acc,
            lr
        ])            

torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

train_loader, test_loader = get_loaders(BATCH_SIZE)
criterion = nn.CrossEntropyLoss()

def evaluate(model):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    with torch.inference_mode():
        for feature, label in test_loader:
            feature, label = feature.to(device), label.to(device)

            outputs = model(feature)
            loss = criterion(outputs, label)

            total_loss += loss.item()
            total += label.shape[0]
            preds = outputs.argmax(dim=1)
            correct += (preds == label).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

    return avg_loss, accuracy

def train_optim(name, lr):
    print(f"\n ------Training with {name}--------")
    model = NN(num_features = 784).to(device)
    optimizer = get_optimizer(name, model, lr)

    train_losses = []
    test_losses = []
    test_accuracies = []

    model.train()

    for epoch in range(EPOCHS):
        total_train_loss = 0.0

        for features, labels in train_loader:

            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_loss)

        test_loss, test_acc = evaluate(model)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        logger(name, epoch+1, avg_loss, test_loss, test_acc, lr)

        print(
                f"{name} | Epoch {epoch+1} | Train Loss {avg_loss:.4f} | Test acc {test_acc:.4f}"

            )
        
    
    return train_losses, test_losses, test_accuracies
    
if __name__ == "__main__":
    results = {}

    for name, lr in EXPS.items():
        if name == "RMSprop":
            results[name] = train_optim(name, lr)

    print("\n Khatam Hogyaa!!!!")
