from demucs import pretrained
from demucs.apply import apply_model
import torchaudio
import torch
import soundfile as sf
import os

# Function to compute the root mean square (RMS) of a tensor (audio track)
def rms(tensor):
    return torch.sqrt(torch.mean(tensor ** 2))

# Function to normalize a track to a target RMS value
def normalize_volume(tensor, target_rms):
    current_rms = rms(tensor)
    scaling_factor = target_rms / current_rms
    return tensor * scaling_factor

# Load the pre-trained Demucs model
model = pretrained.get_model('htdemucs_ft')
model.cuda()  # Use GPU

# Function to convert waveform tensor to WAV file with specified bit depth
def save_as_wav(waveform, sample_rate, output_path, bit_depth=16):
    # Use the correct subtype for 32-bit floating point
    if bit_depth == 32:
        subtype = 'FLOAT'
    else:
        subtype = f'PCM_{bit_depth}'
    
    # Save the waveform as a WAV file
    sf.write(output_path, waveform.cpu().numpy().T, sample_rate, subtype=subtype)


# Load the audio file (replace with your WAV file path)
input_audio = '../music/byebye.wav'
waveform, sr = torchaudio.load(input_audio)

# Check if the audio is mono (1 channel) or stereo (2 channels)
if waveform.shape[0] == 1:
    print("Mono audio detected, converting to stereo.")
    waveform = waveform.repeat(2, 1)  # Duplicate the channel to convert to stereo

# Resample if not 44.1 kHz
target_sr = 44100
if sr != target_sr:
    print(f"Resampling from {sr} Hz to {target_sr} Hz.")
    # Resample the waveform to the target sampling rate
    resampler = torchaudio.transforms.Resample(sr, target_sr)
    waveform = resampler(waveform)

# Add batch dimension: waveform needs to have shape (batch_size, channels, length)
waveform = waveform.unsqueeze(0)

# Move the waveform to the GPU
waveform = waveform.cuda()

# Apply the model to separate the sources on the GPU
sources = apply_model(model, waveform, device='cuda', split=True)

# Check if the sources are correctly separated
if len(sources) == 0:
    raise RuntimeError("Error: The Demucs model did not return any sources.")

# Print the shapes of the sources to understand what we received
for i, source in enumerate(sources[0]):
    print(f"Source {i} shape: {source.shape}")

# Reorder the tracks to match the correct interpretation (as per your corrected order)
bass = sources[0, 1]    # Bass is at index 1
vocals = sources[0, 3]   # Vocals are at index 3
other = sources[0, 2]   # Other instruments (drums) at index 2
drums = sources[0, 0]    # Drums are at index 0

# Define the target bit depth (e.g., 16-bit or 24-bit)
target_bit_depth = 32  # Can be changed to 24 for higher fidelity

# Save each separated source as a WAV file with the correct names (original unnormalized)
save_as_wav(vocals.squeeze(0), target_sr, "vocals_output.wav", bit_depth=target_bit_depth)
save_as_wav(bass.squeeze(0), target_sr, "bass_output.wav", bit_depth=target_bit_depth)
save_as_wav(other.squeeze(0), target_sr, "other_output.wav", bit_depth=target_bit_depth)
save_as_wav(drums.squeeze(0), target_sr, "drums_output.wav", bit_depth=target_bit_depth)

# Scale down drums by 30% (scaling factor of 0.7) but do not normalize them
scaled_drums = drums.squeeze(0) * 0.7
save_as_wav(scaled_drums, target_sr, "scaled_drums_output.wav", bit_depth=target_bit_depth)

print("Original separated tracks and scaled drums saved as WAV files.")

# Set the overall scaling factor to reduce the volume (e.g., 0.7 reduces volume by 30%)
overall_scaling_factor = 0.7

# Calculate the RMS for each track
rms_vocals = rms(vocals.squeeze(0))
rms_bass = rms(bass.squeeze(0))
rms_other = rms(other.squeeze(0))
rms_drums = rms(drums.squeeze(0))

# Find the maximum RMS value among all the tracks (the loudest track)
max_rms = max(rms_vocals, rms_bass, rms_other, rms_drums)

# Normalize each track to match the loudest track's RMS value (max_rms)
normalized_vocals = normalize_volume(vocals.squeeze(0), max_rms)
normalized_bass = normalize_volume(bass.squeeze(0), max_rms)
normalized_other = normalize_volume(other.squeeze(0), max_rms)
normalized_drums = normalize_volume(drums.squeeze(0), max_rms)

# Apply the overall scaling factor to reduce the volume
normalized_vocals *= overall_scaling_factor
normalized_bass *= overall_scaling_factor
normalized_other *= overall_scaling_factor
normalized_drums *= overall_scaling_factor

# Save the normalized and volume-adjusted tracks as WAV files
save_as_wav(normalized_vocals, target_sr, "normalized_vocals_output.wav", bit_depth=target_bit_depth)
save_as_wav(normalized_bass, target_sr, "normalized_bass_output.wav", bit_depth=target_bit_depth)
save_as_wav(normalized_other, target_sr, "normalized_other_output.wav", bit_depth=target_bit_depth)
save_as_wav(normalized_drums, target_sr, "normalized_drums_output.wav", bit_depth=target_bit_depth)

print("Normalization and volume adjustment completed.")

# Combine normalized and unnormalized instrumentals
combined_normalized_instrumental = normalized_drums + normalized_bass + normalized_other
combined_instrumental = drums + bass + other

# Save the combined instrumental tracks
save_as_wav(combined_normalized_instrumental.squeeze(0), target_sr, "combined_normalized_instrumental.wav", bit_depth=target_bit_depth)
save_as_wav(combined_instrumental.squeeze(0), target_sr, "combined_instrumental.wav", bit_depth=target_bit_depth)
