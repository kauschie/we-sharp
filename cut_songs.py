import os
from pydub import AudioSegment
from tqdm import tqdm

# Paths
input_folder = "./p2-data/smaller_test"
output_folder = "./p2-data/smallest_test"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Target segment length in milliseconds
segment_length_ms = 2000  # 2 seconds

# Process all audio files
for file_name in tqdm(os.listdir(input_folder)):
    if file_name.endswith(".wav"):  # Adjust based on formats
        file_path = os.path.join(input_folder, file_name)
        audio = AudioSegment.from_file(file_path).set_frame_rate(24000)  # Ensure 24kHz

        num_segments = len(audio) // segment_length_ms  # Number of 2-sec segments

        for i in range(num_segments):
            start_time = i * segment_length_ms
            end_time = start_time + segment_length_ms
            segment = audio[start_time:end_time]

            # Save each segment
            output_file = os.path.join(output_folder, f"{file_name}_seg{i}.wav")
            segment.export(output_file, format="wav")

print(f"✅ Processed {len(os.listdir(output_folder))} audio segments.")
