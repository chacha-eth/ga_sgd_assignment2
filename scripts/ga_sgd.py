
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

json_file_path = os.path.join('results', "esgd.json")
convergence_speed_plot = os.path.join('figures', "esgd_convergence_speed.png")
stability_plot = os.path.join('figures', "esgd_stability.png")
accuracy_plot = os.path.join('figures', "esgd_accuracy.png")

with open(json_file_path, "r") as file:
    esgd_results = json.load(file)

# Extract the best model per generation
best_models = []

for generation in esgd_results:
    # Find the index of the model with the best (lowest) test loss in this generation
    best_index = generation["test_losses"].index(min(generation["test_losses"]))
    
    # Store the best model's metrics
    best_models.append({
        "Generation": len(best_models),
        "Train Loss": generation["train_losses"][best_index],
        "Train Accuracy": generation["train_accs"][best_index],
        "Test Loss": generation["test_losses"][best_index],
        "Test Accuracy": generation["test_accs"][best_index],
    })

# Convert to DataFrame for analysis
best_models_df = pd.DataFrame(best_models)

# Calculate convergence speed (How fast loss decreases)
best_models_df["Loss Reduction"] = best_models_df["Test Loss"].shift(1) - best_models_df["Test Loss"]
best_models_df["Accuracy Improvement"] = best_models_df["Test Accuracy"] - best_models_df["Test Accuracy"].shift(1)

# Stability (Variance in loss/accuracy over generations)
rolling_window = 10  # Check stability over last 10 generations
best_models_df["Loss Variance"] = best_models_df["Test Loss"].rolling(rolling_window).var()
best_models_df["Accuracy Variance"] = best_models_df["Test Accuracy"].rolling(rolling_window).var()

# Create and save Convergence Speed plot (Loss Reduction per Generation)
plt.figure(figsize=(8, 5))
plt.plot(best_models_df["Generation"], best_models_df["Loss Reduction"], label="Loss Reduction Per Generation", marker='o')
plt.xlabel("Generation")
plt.ylabel("Loss Reduction")
plt.title("Convergence Speed Analysis")
plt.legend()
plt.grid(True)
plt.savefig(convergence_speed_plot)
plt.close()

# Create and save Stability Analysis plot (Variance in Test Loss)
plt.figure(figsize=(8, 5))
plt.plot(best_models_df["Generation"], best_models_df["Loss Variance"], label="Loss Variance")
plt.plot(best_models_df["Generation"], best_models_df["Accuracy Variance"], label="Accuracy Variance")
plt.xlabel("Generation")
plt.ylabel("Variance")
plt.title("Stability Analysis Over Generations")
plt.legend()
plt.grid(True)
plt.savefig(stability_plot)
plt.close()

# Create and save Accuracy Trend plot (Best Accuracy Per Generation)
plt.figure(figsize=(8, 5))
plt.plot(best_models_df["Generation"], best_models_df["Test Accuracy"], label="Best Test Accuracy", marker='o')
plt.xlabel("Generation")
plt.ylabel("Accuracy")
plt.title("Accuracy Improvement Over Generations")
plt.legend()
plt.grid(True)
plt.savefig(accuracy_plot)
plt.close()
