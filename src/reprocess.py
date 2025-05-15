import os
import argparse
import torchaudio

def split_wav_files(input_dir, output_dir, duration, target_sr):
    os.makedirs(output_dir, exist_ok=True)
    target_length = duration * target_sr

    for filename in os.listdir(input_dir):
        if not filename.endswith(".wav"):
            continue

        filepath = os.path.join(input_dir, filename)
        try:
            waveform, sr = torchaudio.load(filepath)

            # Resample if needed
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)

            num_samples = waveform.shape[1]
            num_segments = num_samples // target_length

            for i in range(num_segments):
                segment = waveform[:, i * target_length : (i + 1) * target_length]
                out_filename = f"{os.path.splitext(filename)[0]}_seg{i}.wav"
                out_path = os.path.join(output_dir, out_filename)
                torchaudio.save(out_path, segment, target_sr)

        except Exception as e:
            print(f"[!] Failed to process {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing .wav files")
    parser.add_argument("output_dir", help="Where to save the split .wav files")
    parser.add_argument("--duration", type=int, default=4, help="Target clip duration in seconds (default: 4)")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Target sample rate in Hz (default: 24000)")

    args = parser.parse_args()
    split_wav_files(args.input_dir, args.output_dir, args.duration, args.sample_rate)

