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

# Function to convert waveform tensor to mp3 file
def save_as_mp3(waveform, sample_rate, output_path):
    temp_wav = "temp_output.wav"
    sf.write(temp_wav, waveform.cpu().numpy().T, sample_rate)  # Save as .wav first
    os.system(f"ffmpeg -i {temp_wav} -q:a 2 {output_path}")  # Convert wav to mp3 using ffmpeg
    os.remove(temp_wav)  # Remove temp wav file

# Load the audio file (replace with your mp3 path)
input_audio = '../music/itsme.mp3'
waveform, sr = torchaudio.load(input_audio)


# Check if the audio is mono (1 channel) or stereo (2 channels)
if waveform.shape[0] == 1:
    print("Mono audio detected, converting to stereo.")
    waveform = waveform.repeat(2, 1)  # Duplicate the channel to convert to stereo

# resampe if not 44100 khz
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

# Save each separated source as an MP3 file with the correct names (original unnormalized)
save_as_mp3(vocals.squeeze(0), sr, "vocals_output.mp3")
save_as_mp3(bass.squeeze(0), sr, "bass_output.mp3")
save_as_mp3(other.squeeze(0), sr, "other_output.mp3")
save_as_mp3(drums.squeeze(0), sr, "drums_output.mp3")

# Scale down drums by 30% (scaling factor of 0.7) but do not normalize them
scaled_drums = drums.squeeze(0) * 0.7
save_as_mp3(scaled_drums, sr, "scaled_drums_output.mp3")

print("Original separated tracks and scaled drums saved as MP3 files.")

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

# Save the normalized and volume-adjusted tracks as MP3 files
save_as_mp3(normalized_vocals, sr, "normalized_vocals_output.mp3")
save_as_mp3(normalized_bass, sr, "normalized_bass_output.mp3")
save_as_mp3(normalized_other, sr, "normalized_other_output.mp3")
save_as_mp3(normalized_drums, sr, "normalized_drums_output.mp3")

print("Normalization and volume adjustment completed.")


combined_normalized_instrumental = normalized_drums + normalized_bass + normalized_other
combined_instrumental = drums + bass + other
save_as_mp3(combined_normalized_instrumental.squeeze(0), sr, "combined_normalized_instrumental.mp3")
save_as_mp3(combined_instrumental.squeeze(0), sr, "combined_instrumental.mp3")