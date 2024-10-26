"""
Example Usage:

## CUDA (GPU) Enabled
python demucs_test.py your_audio_file.wav --device cuda  

## CPU Enabled
python demucs_test.py your_audio_file.wav --device cpu

## without specifying
## this will prompt during execution
python demucs_test.py your_audio_file.wav
"""

import torchaudio
import torch
import os
import argparse
from demucs import pretrained
from demucs.apply import apply_model

# Function to compute the root mean square (RMS) of a tensor (audio track)
def rms(tensor):
    return torch.sqrt(torch.mean(tensor ** 2))

# Function to normalize a track to a target RMS value
def normalize_volume(tensor, target_rms):
    current_rms = rms(tensor)
    scaling_factor = target_rms / current_rms
    return tensor * scaling_factor

# Function to save waveform as WAV or MP3 file using torchaudio
# Function to save waveform as WAV or MP3 file using torchaudio
def save_audio(waveform, sample_rate, output_path, output_format="wav"):
    # Move the waveform to CPU before saving
    waveform = waveform.cpu()
    
    if output_format == "wav":
        torchaudio.save(output_path, waveform, sample_rate)
    elif output_format == "mp3":
        torchaudio.save(output_path, waveform, sample_rate, format="mp3")


# Function to resample audio if necessary
def resample_audio(waveform, sr, target_sr=44100):
    if sr != target_sr:
        print(f"Resampling from {sr} Hz to {target_sr} Hz.")
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        return resampler(waveform), target_sr
    return waveform, sr

# Function to convert mono to stereo if necessary
def convert_to_stereo(waveform):
    if waveform.shape[0] == 1:
        print("Mono audio detected, converting to stereo.")
        waveform = waveform.repeat(2, 1)
    return waveform

# Function to load the input file (MP3 or WAV)
def load_audio(file_path):
    """
    Function to load the audio file (MP3 or WAV) with proper validation.
    - Checks if the file exists.
    - Verifies if it's a valid MP3 or WAV file.
    - Converts mono to stereo if needed.
    - Resamples to the target sample rate if needed.
    """
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")
    
    # Detect file type
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.mp3':
        print("MP3 file detected.")
        output_format = 'mp3'
    elif ext == '.wav':
        print("WAV file detected.")
        output_format = 'wav'
    else:
        raise ValueError("Unsupported file format. Please provide an MP3 or WAV file.")
    
    # Attempt to load the audio file
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        raise RuntimeError(f"Error loading audio file '{file_path}': {e}")
    
    # Validate audio properties (e.g., channels, length)
    if waveform.shape[1] == 0:
        raise ValueError(f"Error: The audio file '{file_path}' contains no audio data.")
    
    if waveform.shape[0] > 2:
        raise ValueError(f"Error: The audio file '{file_path}' has more than 2 channels. Stereo (2 channels) or mono (1 channel) expected.")
    
    # Convert mono to stereo if needed
    waveform = convert_to_stereo(waveform)

    # Resample if needed
    waveform, sr = resample_audio(waveform, sr)
    
    return waveform, sr, output_format

# Function to process and separate sources using Demucs
def separate_sources(waveform, sr, model, device):
    waveform = waveform.unsqueeze(0).to(device)  # Add batch dimension and move to device (GPU or CPU)
    sources = apply_model(model, waveform, device=device, split=True)
    if len(sources) == 0:
        raise RuntimeError("Error: The Demucs model did not return any sources.")
    
    return sources[0]  # Return separated sources

# Function to save separated and normalized tracks
def save_tracks(sources, sr, output_format, output_prefix="output"):
    # Reorder sources to the correct interpretation
    bass, vocals, other, drums = sources[1], sources[3], sources[2], sources[0]

    # Save the original separated tracks
    # save_audio(vocals.squeeze(0), sr, f"{output_prefix}_vocals.{output_format}", output_format)
    save_audio(bass.squeeze(0), sr, f"{output_prefix}_bass.{output_format}", output_format)
    save_audio(other.squeeze(0), sr, f"{output_prefix}_other.{output_format}", output_format)
    save_audio(drums.squeeze(0), sr, f"{output_prefix}_drums.{output_format}", output_format)

    # Scale down drums by 30% but do not normalize them
    # scaled_drums = drums.squeeze(0) * 0.7
    # save_audio(scaled_drums, sr, f"{output_prefix}_scaled_drums.{output_format}", output_format)

    # Normalize and adjust volume of each track
    # max_rms = max(rms(vocals.squeeze(0)), rms(bass.squeeze(0)), rms(other.squeeze(0)), rms(drums.squeeze(0)))

    # normalized_vocals = normalize_volume(vocals.squeeze(0), max_rms)
    # normalized_bass = normalize_volume(bass.squeeze(0), max_rms)
    # normalized_other = normalize_volume(other.squeeze(0), max_rms)
    # normalized_drums = normalize_volume(drums.squeeze(0), max_rms)

    # overall_scaling_factor = 0.7
    # normalized_vocals *= overall_scaling_factor
    # normalized_bass *= overall_scaling_factor
    # normalized_other *= overall_scaling_factor
    # normalized_drums *= overall_scaling_factor

    # Save normalized tracks
    # save_audio(normalized_vocals, sr, f"{output_prefix}_normalized_vocals.{output_format}", output_format)
    # save_audio(normalized_bass, sr, f"{output_prefix}_normalized_bass.{output_format}", output_format)
    # save_audio(normalized_other, sr, f"{output_prefix}_normalized_other.{output_format}", output_format)
    # save_audio(normalized_drums, sr, f"{output_prefix}_normalized_drums.{output_format}", output_format)

    # Combine normalized and unnormalized instrumentals and save
    combined_instrumental = drums + bass + other
    # combined_vocal_drum = vocals + scaled_drums
    # combined_other_base_scaled_drum = other + bass + scaled_drums
    # combined_other_base = other + bass

    # save_audio(combined_normalized_instrumental.squeeze(0), sr, f"{output_prefix}_combined_normalized_instrumental.{output_format}", output_format)
    save_audio(combined_instrumental.squeeze(0), sr, f"{output_prefix}_combined_instrumental.{output_format}", output_format)
    # save_audio(combined_vocal_drum.squeeze(0), sr, f"{output_prefix}_combined_vocal_drum.{output_format}", output_format)
    # save_audio(combined_other_base_scaled_drum.squeeze(0), sr, f"{output_prefix}_combined_other_base_scaled_drum.{output_format}", output_format)
    # save_audio(combined_other_base.squeeze(0), sr, f"{output_prefix}_combined_other_base.{output_format}", output_format)
    # save_audio(other.squeeze(0), sr, f"{output_prefix}_other.{output_format}", output_format)

    print("Finished saving test tracks")

# Function to check for CUDA and prompt the user if needed
def choose_device(device=None):
    if device:
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available on this machine. Using CPU instead.")
            return 'cpu'
        elif device in ['cuda', 'cpu']:
            return device
        else:
            raise ValueError("Invalid device argument. Please use 'cuda' or 'cpu'.")
    else:
        if torch.cuda.is_available():
            while True:
                try:
                    response = input("CUDA is available. Would you like to use it? [y/n]: ").strip().lower()
                    if response == 'y':
                        return 'cuda'
                    elif response == 'n':
                        return 'cpu'
                    else:
                        print("Invalid response. Please type 'y' for yes or 'n' for no.")
                except (KeyboardInterrupt, EOFError):
                    print("\nUser interrupted the input. Defaulting to CPU.")
                    return 'cpu'
        else:
            print("CUDA is not available. Using CPU.")
            return 'cpu'

# Main function to handle the full process
def main(file_path, device=None):
    # Choose the device (CUDA or CPU)
    device = choose_device(device)

    # Load the pre-trained Demucs model
    model = pretrained.get_model('htdemucs_ft')
    model.to(device)  # Move model to selected device

    # Load and preprocess the input audio file
    waveform, sr, output_format = load_audio(file_path)

    # Separate sources using Demucs
    sources = separate_sources(waveform, sr, model, device)

    # Save tracks (original, normalized, combined)
    save_tracks(sources, sr, output_format)

    
# Entry point for the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio file (MP3 or WAV) and separate sources using Demucs.")
    parser.add_argument("file_path", type=str, help="Path to the input MP3 or WAV file.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use for processing ('cuda' or 'cpu').")
    args = parser.parse_args()

    main(args.file_path, args.device)
