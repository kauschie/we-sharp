import torchaudio
import os

input_folder = "p2-data/smallest_test_24k"
output_folder = "p2-data/smallest_test_16khz"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".wav"):  # Only process WAV files
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        # Load and resample audio
        waveform, sample_rate = torchaudio.load(input_path)
        resampled_waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        # Save the resampled file
        torchaudio.save(output_path, resampled_waveform, 16000)

print("✅ All files converted to 16 kHz.")
