import sounddevice as sd
import numpy as np
import soundfile as sf
from transformers import EncodecModel, AutoProcessor
import time
import torch

def load_audio_file(file_path, target_sampling_rate=24000):
    """Load and resample audio file if needed"""
    audio_data, sampling_rate = sf.read(file_path)
    
    # Convert stereo to mono by averaging channels if needed
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Normalize audio to be between -1 and 1
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    return audio_data, sampling_rate

def play_audio(audio_data, sampling_rate):
    """Play audio using sounddevice"""
    audio_data = np.clip(audio_data, -1, 1)
    print("Playing audio...")
    sd.play(audio_data, sampling_rate)
    sd.wait()

# Hardcode your audio path here
AUDIO_PATH = "../assets/output_other.mp3"

# Load model
print("Loading model and processor...")
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# Load and process the audio
print("Loading audio file...")
audio_sample, original_sampling_rate = load_audio_file(AUDIO_PATH)
sampling_rate = processor.sampling_rate

print(f"Original audio: {len(audio_sample)} samples, {original_sampling_rate}Hz")
print(f"Duration: {len(audio_sample)/original_sampling_rate:.2f} seconds")

# Process through model
print("Processing audio through model...")
inputs = processor(raw_audio=audio_sample, sampling_rate=sampling_rate, return_tensors="pt")
encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
decoded_audio = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
decoded_audio_np = decoded_audio.detach().squeeze().numpy()

# Calculate compression ratio
original_size = audio_sample.shape[0]
compressed_size = encoder_outputs.audio_codes.shape[-1] * encoder_outputs.audio_codes.shape[-2]
compression_ratio = original_size / compressed_size

print(f"\nCompression ratio: {compression_ratio:.1f}:1")
print(f"Original samples: {original_size}")
print(f"Compressed tokens: {compressed_size}")

# Interactive playback
while True:
    choice = input("\nWhat would you like to do?\n1: Play original audio\n2: Play reconstructed audio\n3: Play both (with pause between)\n4: Save files\n5: Show compression details\n6: Exit\n> ")
    
    if choice == '1':
        play_audio(audio_sample, sampling_rate)
    elif choice == '2':
        play_audio(decoded_audio_np, sampling_rate)
    elif choice == '3':
        print("Playing original audio...")
        play_audio(audio_sample, sampling_rate)
        time.sleep(1)  # 1 second pause
        print("Playing reconstructed audio...")
        play_audio(decoded_audio_np, sampling_rate)
    elif choice == '4':
        orig_filename = "original_audio.wav"
        recon_filename = "reconstructed_audio.wav"
        sf.write(orig_filename, audio_sample, sampling_rate)
        sf.write(recon_filename, decoded_audio_np, sampling_rate)
        print(f"Saved audio files as '{orig_filename}' and '{recon_filename}'")
    elif choice == '5':
        print("\nCompression Details:")
        print(f"Original audio length: {len(audio_sample)} samples ({len(audio_sample)/sampling_rate:.2f} seconds)")
        print(f"Compressed representation: {encoder_outputs.audio_codes.shape}")
        print(f"Compression ratio: {compression_ratio:.1f}:1")
        print(f"Number of unique tokens used: {len(torch.unique(encoder_outputs.audio_codes))}")
    elif choice == '6':
        break
    else:
        print("Invalid choice. Please try again.")