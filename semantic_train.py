import os
import matplotlib
import matplotlib.pyplot as plt
import torch
from audiolm_pytorch import HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer

# Use the Agg backend for headless environments
matplotlib.use('Agg')

# Paths to models and dataset
hubert_checkpoint_path = './models/hubert_base_ls960.pt'
hubert_kmeans_path = './models/hubert_base_ls960_L9_km500.bin'
dataset_path = './dbo'
results_folder = './results'  # Results directory
save_path = os.path.join(results_folder, 'semantic_transformer_best.pth')

# Initialize HubertWithKmeans
wav2vec = HubertWithKmeans(
    checkpoint_path=hubert_checkpoint_path,
    kmeans_path=hubert_kmeans_path
)

wav2vec = wav2vec.cuda()

# Define and initialize the Semantic Transformer
semantic_transformer = SemanticTransformer(
    num_semantic_tokens=wav2vec.codebook_size,  # From HubertWithKmeans
    dim=1024,  # Transformer dimensionality
    depth=6,  # Number of transformer layers
    flash_attn=True  # Use Flash Attention for efficiency
)
semantic_transformer = semantic_transformer.cuda()

# Trainer for the Semantic Transformer
semantic_trainer = SemanticTransformerTrainer(
    transformer=semantic_transformer,
    wav2vec=wav2vec,  # HubertWithKmeans model
    folder=dataset_path,  # Path to your training data
    batch_size=4,  # Adjust based on GPU memory
    grad_accum_every=8,  # Gradient accumulation steps
    data_max_length=240000,  # Max number of audio samples (24 kHz * 10 seconds)
    num_train_steps=1000,  # Increase for longer training
    results_folder=results_folder  # Specify custom results folder
)

# Train and track losses
def train_and_plot():
    training_losses = []
    best_train_loss = float('inf')

    print("Starting training for the Semantic Transformer...")

    for step in range(semantic_trainer.num_train_steps):
        train_step_result = semantic_trainer.train_step()  # Returns a dict
        if not isinstance(train_step_result, dict) or 'loss' not in train_step_result:
            raise ValueError(f"Unexpected train_step result: {train_step_result}")

        train_loss = train_step_result['loss']  # Extract loss value
        if not isinstance(train_loss, (float, int)):
            raise ValueError(f"Unexpected train_loss type: {type(train_loss)} - {train_loss}")

        training_losses.append(train_loss)

        # # Save the best model based on training loss
        # if train_loss < best_train_loss:
        #     best_train_loss = train_loss
        #     print(f"Saving best model at step {step} with training loss {train_loss:.4f}")
        #     torch.save(semantic_transformer.state_dict(), save_path)

    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss')

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)

    # Save the plot to the results folder
    plot_path = os.path.join(results_folder, 'loss_plot.png')
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}.")

# Execute training and plotting
train_and_plot()

print(f"Best Semantic Transformer saved to {save_path}")
