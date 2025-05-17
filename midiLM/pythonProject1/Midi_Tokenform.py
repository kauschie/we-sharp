import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Input & Output Folders ===
MIDI_TXT_FOLDER = r"../melody_only_output"      # Input: cleaned MIDI text files
TOKEN_TXT_FOLDER = r"../TOKEN_TXT"              # Output: pitch-shifted token files
SLIDING_FOLDER = r"../TOKEN_SLIDING"            # Output: sliding window segments
MLP_FOLDER = r"../MLP_TXT"                      # Optional output (unused in this script)

# === Sliding Window Parameters ===
WINDOW_SIZE = 1024
OVERLAP = 512
MIN_LAST_WINDOW = 796  # Minimum size for the last window to be considered valid

# === Note Pitch Shift Offsets ===
offsets = [0, -1, 1]  # Original pitch, pitch -1, pitch +1

# === Ensure output folders exist ===
os.makedirs(TOKEN_TXT_FOLDER, exist_ok=True)
os.makedirs(SLIDING_FOLDER, exist_ok=True)
os.makedirs(MLP_FOLDER, exist_ok=True)

def convert_txt_to_tokens(input_txt, output_base):
    """
    Convert MIDI text file to 3 token files with different pitch offsets.
    Each line is transformed into a token format: [on/off]_[pitch]_[delta_time].
    """
    output_files = {i: open(f"{output_base}_{i}.txt", "w") for i in range(3)}
    with open(input_txt, "r") as infile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            event_type = parts[0]
            if event_type not in ["note_on", "note_off"]:
                continue
            note = int(parts[1])
            velocity = int(parts[2])
            delta_time = int(parts[3])
            token_event = "on" if event_type == "note_on" else "off"

            for i, offset in enumerate(offsets):
                new_note = max(21, min(109, note + offset))  # Clamp pitch between 21–109
                output_files[i].write(f"{token_event}_{new_note}_{delta_time}\n")

    for f in output_files.values():
        f.close()

def slide_window_and_save(input_txt, output_folder):
    """
    Divide a token file into fixed-length overlapping sliding windows and save each segment.
    """
    with open(input_txt, "r") as f:
        lines = f.readlines()

    if len(lines) <= 1:
        return 0

    total_lines = len(lines)
    base_filename = os.path.basename(input_txt).replace(".txt", "")
    output_files = []
    start_idx = 0
    file_num = 1

    # Main sliding window loop
    while start_idx + WINDOW_SIZE <= total_lines:
        end_idx = start_idx + WINDOW_SIZE
        output_filename = os.path.join(output_folder, f"{base_filename}_{file_num}.txt")
        with open(output_filename, "w") as out_f:
            out_f.writelines(lines[start_idx:end_idx])
        output_files.append(output_filename)
        file_num += 1
        start_idx += WINDOW_SIZE - OVERLAP

    # Handle the final window if it's sufficiently long
    if total_lines - start_idx >= MIN_LAST_WINDOW:
        output_filename = os.path.join(output_folder, f"{base_filename}_{file_num}.txt")
        with open(output_filename, "w") as out_f:
            out_f.writelines(lines[-WINDOW_SIZE:])
        output_files.append(output_filename)

    return len(output_files)

def process_single_midi(midi_file):
    """
    Process a single MIDI file:
    1. Convert to token format (3 pitch-shifted versions)
    2. Apply sliding window segmentation to each
    """
    input_path = os.path.join(MIDI_TXT_FOLDER, midi_file)
    base_token_path = os.path.join(TOKEN_TXT_FOLDER, midi_file.replace(".txt", ""))
    convert_txt_to_tokens(input_path, base_token_path)

    total = 0
    for i in range(3):  # Process each pitch-shifted token file
        token_file = f"{base_token_path}_{i}.txt"
        total += slide_window_and_save(token_file, SLIDING_FOLDER)
    return total

def process_all_midis(max_workers=None):
    """
    Process all MIDI text files in the input folder using multithreading.
    """
    midi_files = [f for f in os.listdir(MIDI_TXT_FOLDER) if f.endswith(".txt")]
    print(f"🎵 Starting to process {len(midi_files)} MIDI files...\n")

    total_sliding_files = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_midi, midi): midi for midi in midi_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="file"):
            total_sliding_files += future.result()

    print(f"\n✅ Processing complete! Total sliding windows generated: {total_sliding_files}")

# ✅ Entry point
if __name__ == "__main__":
    process_all_midis(max_workers=24)  # Adjust worker count as needed
