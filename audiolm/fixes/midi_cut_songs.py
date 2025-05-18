import os
from pydub import AudioSegment
from pydub.utils import make_chunks
from tqdm import tqdm

# Parameters
segment_length_ms = 2000  # 2 seconds
rms_threshold = 100  # Adjust this if needed after testing (typical range 50–200)

# Paths
input_folder = "hz_16k_wav"
output_folder_16k = "hz_2s_16k"
output_folder_24k = "hz_2s_24k"
log_file = "blank_segments_log.txt"

# Ensure output folders exist
os.makedirs(output_folder_16k, exist_ok=True)
os.makedirs(output_folder_24k, exist_ok=True)

# Open log file for blank segments
with open(log_file, "w") as blank_log:
    blank_log.write("filename,reason,details\n")

    # Process files with tqdm progress bar
    for file_name in tqdm(os.listdir(input_folder), desc="Cutting audio"):
        if not file_name.lower().endswith(".wav"):
            continue

        file_path = os.path.join(input_folder, file_name)
        audio = AudioSegment.from_file(file_path)

        if len(audio) < segment_length_ms:
            blank_log.write(f"{file_name},skipped_entirely,too_short\n")
            continue

        chunks = make_chunks(audio, segment_length_ms)

        for i, chunk in enumerate(chunks):
            if chunk.rms < rms_threshold:
                blank_log.write(f"{file_name}_seg{i}.wav,blank_segment,rms={chunk.rms}\n")
                continue

            # Save 16kHz version
            output_file_16k = os.path.join(output_folder_16k, f"{file_name}_seg{i}.wav")
            chunk.set_frame_rate(16000).export(output_file_16k, format="wav")

            # Save 24kHz version
            output_file_24k = os.path.join(output_folder_24k, f"{file_name}_seg{i}.wav")
            chunk.set_frame_rate(24000).export(output_file_24k, format="wav")

print(f"\n✅ Done! Blank segments logged to '{log_file}'")
