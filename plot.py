import os
import csv
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

def load_csv(filepath):
    epochs = []
    train_loss = []
    test_loss = []
    test_acc = []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            test_loss.append(float(row["test_loss"]))
            test_acc.append(float(row["test_acc"]))

        return epochs, train_loss, test_loss, test_acc
    
    plt.figure()

for file in os.listdir(RESULTS_DIR):
    if file.endswith(".csv"):
        optimizer = file.replace(".csv", "")
        epochs, train_loss,_ , _ = load_csv(os.path.join(RESULTS_DIR, file))

        plt.plot(epochs, train_loss, label = optimizer)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.yscale("log")
plt.legend()
plt.show()


plt.figure()
for file in os.listdir(RESULTS_DIR):
    if file.endswith(".csv"):
        optimizer = file.replace(".csv", "")
        epochs, _, _, test_acc = load_csv(os.path.join(RESULTS_DIR,file))
        plt.plot(epochs, test_acc, label = optimizer)
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Epochs")
plt.yscale("log")
plt.legend()
plt.show()