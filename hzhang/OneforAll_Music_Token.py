import os
import librosa
import csv
import numpy as np
import torch
from transformers import EncodecModel, AutoProcessor
from scipy.signal import resample


# -------------------- Load Encodec Model --------------------
def load_encodec_model(target_bandwidth=6):
    """
    Load Encodec model and processor with the specified target bandwidth.
    """
    model = EncodecModel.from_pretrained("facebook/encodec_24khz", target_bandwidths=[target_bandwidth])
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    return model, processor


# -------------------- Music Segmentation (without saving files) --------------------
def split_music(y, sr, timestamps):
    """
    Splits the input audio signal based on beat timestamps.

    Args:
        y: Audio waveform.
        sr: Sample rate of the audio.
        timestamps: List of (time, beat) tuples from the CSV file.

    Returns:
        A segmented portion of the audio based on beats or None if invalid.
    """
    first_beat_index = next((i for i, t in enumerate(timestamps) if t[1] == 1), None)
    last_beat_index = next((i for i, t in reversed(list(enumerate(timestamps))) if t[1] == 1), None)

    # If there are no valid beats, return None
    if first_beat_index is None or last_beat_index is None or first_beat_index >= last_beat_index:
        return None

        # Limit to a maximum of 161 beats (to ensure proper segmentation)
    if last_beat_index - first_beat_index + 1 > 161:
        last_beat_index = next((i for i in reversed(range(first_beat_index, last_beat_index + 1))
                                if i - first_beat_index + 1 <= 161 and timestamps[i][1] == 1), None)

    # Extract start and end times from timestamps
    start_time = timestamps[first_beat_index][0]
    end_time = timestamps[last_beat_index][0]

    # Convert time to sample indices
    start_index = int(sr * start_time)
    end_index = int(sr * end_time)

    return y[start_index:end_index]


# -------------------- Encode Music to Tokens (No Intermediate File) --------------------
def encode_music(model, processor, audio_sample, sample_rate, output_folder, filename):
    """
    Uses Encodec to convert the audio sample into tokens and stores them in text files.

    Args:
        model: Encodec model for processing.
        processor: Audio processor for Encodec.
        audio_sample: The segmented waveform.
        sample_rate: The sample rate of the audio.
        output_folder: The destination folder for tokenized output.
        filename: Name of the original file for saving tokenized output.
    """
    # Convert stereo to mono if needed
    if len(audio_sample.shape) > 1:
        audio_sample = audio_sample.mean(axis=1)

    # Resample the audio to 24000 Hz
    num_samples = int(len(audio_sample) * 24000 / sample_rate)
    audio_sample = resample(audio_sample, num_samples)

    # Process input for the Encodec model
    inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

    # Encode the audio sample
    encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
    audio_codes = encoder_outputs.audio_codes
    audio_s = audio_codes[0].squeeze()

    # Save tokenized output to text files
    for i in range(len(audio_s)):
        folder_name = os.path.join(output_folder, f"MusicTxT_{i}")
        os.makedirs(folder_name, exist_ok=True)

        txt_path = os.path.join(folder_name, f"{os.path.splitext(filename)[0]}_{i}.txt")
        listout = audio_s[i].tolist()
        with open(txt_path, "w") as f:
            f.write(" ".join(map(str, listout)))
        print(f"Tokens saved to {txt_path}")


# -------------------- Main Processing Function --------------------
def process_music(music_folder, csv_folder, output_folder, target_bandwidth=6):
    """
    Main function to process all music files:
    - Reads audio files
    - Splits them based on beat timestamps
    - Encodes them into tokens using Encodec

    Args:
        music_folder: Path to the directory containing .wav files.
        csv_folder: Path to the directory containing corresponding beat timestamp CSV files.
        output_folder: Path where the tokenized outputs will be stored.
        target_bandwidth: The bandwidth setting for the Encodec model.
    """
    # Load the Encodec model once for efficiency
    model, processor = load_encodec_model(target_bandwidth)

    # Iterate through all audio files in the folder
    for filename in os.listdir(music_folder):
        if filename.endswith(".wav"):
            music_path = os.path.join(music_folder, filename)
            csv_path = os.path.join(csv_folder, os.path.splitext(filename)[0] + ".csv")

            # Skip if there is no corresponding CSV file
            if not os.path.exists(csv_path):
                print(f"CSV file not found for: {filename}")
                continue

            # Load the audio file
            y, sr = librosa.load(music_path, sr=None)

            # Read beat timestamps from the CSV file
            timestamps = []
            with open(csv_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    time = float(row[0])
                    beat = int(row[1])
                    timestamps.append((time, beat))

            # Skip if the CSV file has too few timestamps
            if len(timestamps) < 2:
                print(f"{filename} CSV does not have enough rows.")
                continue

            # Extract a segment of the audio based on the beat timestamps
            segment = split_music(y, sr, timestamps)
            if segment is None:
                print(f"{filename} does not have valid beat ranges.")
                continue

            # Directly encode the segment into tokens (without saving as an intermediate file)
            encode_music(model, processor, segment, sr, output_folder, filename)


# -------------------- Script Entry Point --------------------
if __name__ == "__main__":
    process_music("testMusic", "testOutput", "MusicTxT", target_bandwidth=6)
