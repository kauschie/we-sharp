import matplotlib
import matplotlib.pyplot as plt
import os

# Use the Agg backend for headless environments
matplotlib.use('Agg')  # Prevents Qt-related issues in headless setups

# Directory to save the plot
results_folder = './results'
os.makedirs(results_folder, exist_ok=True)

# Example data
training_losses = [6.0, 5.5, 4.8, 4.2, 3.9, 3.5, 3.2, 2.9, 2.5, 2.2]
validation_losses = [5.8, 5.3, 4.9, 4.3, 4.0, 3.7, 3.4, 3.1, 2.8, 2.5]

# Plot the loss curves
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss', linestyle='--')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.grid(True)

# Save the plot to the results folder
plot_path = os.path.join(results_folder, 'debug_loss_plot.png')
plt.savefig(plot_path)
print(f"Loss plot saved to {plot_path}.")
