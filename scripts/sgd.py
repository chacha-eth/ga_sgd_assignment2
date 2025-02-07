# Re-import necessary libraries since execution state was reset
import pandas as pd
import matplotlib.pyplot as plt
import os


csv_file=os.path.join('results', "baseline.csv")
# Convert the data string into a DataFrame
from io import StringIO
# Load data from CSV file (Assuming user will provide the correct file path)
df = pd.read_csv(csv_file)

# Define save paths
loss_plot_path = os.path.join('figures', "sgd_convergence_speed.png")
accuracy_plot_path = os.path.join('figures', "sgd_stability_accuracy.png")
# Plot Train and Test Loss for Convergence Speed Analysis
# Select epochs to display (5, 10, 20, ..., 100)
display_epochs = list(range(5, 101, 10))
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_losses"], label="Train Loss")
plt.plot(df["epoch"], df["test_losses"], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train & Test Loss (Convergence Speed)")
plt.xticks(display_epochs) 
plt.legend()
plt.grid(True)
plt.savefig(loss_plot_path)
plt.close()

# Plot Train and Test Accuracy for Stability & Accuracy Analysis
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_accuracies"], label="Train Accuracy")
plt.plot(df["epoch"], df["test_accuracies"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train & Test Accuracy (Stability & Generalization)")
plt.xticks(display_epochs) 
plt.legend()
plt.grid(True)
plt.savefig(accuracy_plot_path)
plt.close()

