import os
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from audiolm_pytorch import HubertWithKmeans

# Define Dataset Class
class AudioDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # Use only one specific file for debugging
        # self.file_paths = [os.path.join(dataset_path, 'Aerosmith - Crazy_dbo_cut1.wav')]  # Replace 'specific_file.wav' with your filename
        # Original code to process all files (commented out for now)
        self.file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.wav')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio, sample_rate = torchaudio.load(self.file_paths[idx])
        return audio, sample_rate, self.file_paths[idx]  # Return file path for logging

# Initialize HubertWithKmeans
wav2vec = HubertWithKmeans(
    checkpoint_path='./models/hubert_base_ls960.pt',  # Update with correct path
    kmeans_path='./models/hubert_base_ls960_L9_km500.bin'
).cuda()  # Use GPU

# Path to preprocessed audio files
dataset_path = "./dbo"
audio_dataset = AudioDataset(dataset_path)
data_loader = DataLoader(audio_dataset, batch_size=1, shuffle=False)

# Function to calculate token lengths
def calculate_token_lengths(data_loader, wav2vec):
    token_lengths = []
    file_paths = []
    for audio, sample_rate, file_path in data_loader:
        try:
            # Ensure audio tensor has shape [time_steps]
            audio = audio.squeeze(0)  # Remove batch dimension added by torchaudio.load

            # Resample to 16kHz if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate.item(), new_freq=16000)
                audio = resampler(audio)

            # Debugging: Print the shape of the input tensor
            # print(f"Input shape to model: {audio.shape}")

            # Extract tokens
            with torch.no_grad():
                tokens = wav2vec(audio.cuda())  # Move to GPU
            token_lengths.append(tokens.shape[-1])
            file_paths.append(file_path[0])

            print(f"File: {file_path[0]}, Tokens: {tokens.shape[-1]}")
        except Exception as e:
            print(f"Error processing file {file_path[0]}: {e}")
    return file_paths, token_lengths

# Calculate token lengths
file_paths, token_lengths = calculate_token_lengths(data_loader, wav2vec)

# Print statistics
print(f"Max token length: {max(token_lengths)}")
print(f"Min token length: {min(token_lengths)}")
print(f"Average token length: {sum(token_lengths) / len(token_lengths):.2f}")
