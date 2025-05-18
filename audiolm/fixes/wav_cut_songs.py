import os
from pydub import AudioSegment
from tqdm import tqdm

# Paths
input_folder = "../p2-data/processed_wav"
output_folder_16k = "../p2-data/p2_4s_16k"
output_folder_24k = "../p2-data/p2_4s_24k"

# Ensure output folders exist
os.makedirs(output_folder_16k, exist_ok=True)
os.makedirs(output_folder_24k, exist_ok=True)

# Target segment length in milliseconds
segment_length_ms = 2000  # 4 seconds

# Process all audio files
for file_name in tqdm(os.listdir(input_folder)): # tqdm creates a progress bar
    if file_name.endswith(".wav"):  
        file_path = os.path.join(input_folder, file_name)
        audio = AudioSegment.from_file(file_path)

        num_segments = len(audio) // segment_length_ms  # Number of 4-sec segments

        for i in range(num_segments):
            start_time = i * segment_length_ms
            end_time = start_time + segment_length_ms
            segment = audio[start_time:end_time]

            # Save 16kHz version (for BERT)
            output_file_16k = os.path.join(output_folder_16k, f"{file_name}_seg{i}.wav")
            segment_16k = segment.set_frame_rate(16000)
            segment_16k.export(output_file_16k, format="wav")

            # Save 24kHz version (for EnCodec)
            output_file_24k = os.path.join(output_folder_24k, f"{file_name}_seg{i}.wav")
            segment_24k = segment.set_frame_rate(24000)
            segment_24k.export(output_file_24k, format="wav")

print(f"✅ Processed {len(os.listdir(output_folder_16k))} 16kHz audio segments.")
print(f"✅ Processed {len(os.listdir(output_folder_24k))} 24kHz audio segments.")
