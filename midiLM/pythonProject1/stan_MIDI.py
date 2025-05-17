import os
import mido
from collections import defaultdict
from tqdm import tqdm  # Progress bar library

# === Folder paths ===
INPUT_FOLDER = r"TestMidi"
OUTPUT_FOLDER = r"../MIDI_TXT"

# === Ensure output folder exists ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Data statistics containers ===
velocity_count = defaultdict(int)  # Frequency of note velocities
note_count = defaultdict(int)      # Frequency of note pitches
total_events = 0                   # Total number of MIDI events

def midi_to_text(input_midi, output_txt):
    """
    Convert a MIDI file to plain text format.
    Only 'note_on' and 'note_off' events are preserved.
    """
    global total_events

    midi_file = mido.MidiFile(input_midi)

    with open(output_txt, "w") as f:
        for track in midi_file.tracks:
            for msg in track:
                try:
                    total_events += 1  # Count every MIDI message

                    # Only process note_on and note_off events
                    if msg.type in ["note_on", "note_off"]:
                        event_data = [msg.type, msg.channel, msg.note, msg.velocity, msg.time]

                        # Track velocity and note frequency (only for note_on with velocity > 0)
                        if msg.type == "note_on" and msg.velocity > 0:
                            velocity_count[msg.velocity] += 1
                            note_count[msg.note] += 1

                        # Write the event to the output file (space-separated)
                        f.write(" ".join(map(str, event_data)) + "\n")

                except Exception as e:
                    print(f"⚠️ Error processing message: {e}")
                    print(f"❌ Problematic MIDI message: {msg}")
                    continue  # Skip malformed messages without stopping execution

def process_all_midis(input_folder, output_folder):
    """
    Traverse all MIDI files in the input folder, convert them to text,
    and collect statistical data on note usage and velocities.
    """
    midi_files = [f for f in os.listdir(input_folder) if f.endswith(".mid")]
    total_midi_count = len(midi_files)

    print(f"🎵 Starting conversion of {total_midi_count} MIDI files...\n")

    with tqdm(total=total_midi_count, desc="Conversion Progress", unit="file") as pbar:
        for i, midi_file in enumerate(midi_files):
            input_path = os.path.join(input_folder, midi_file)
            output_txt = os.path.join(output_folder, midi_file.replace(".mid", ".txt"))

            midi_to_text(input_path, output_txt)
            pbar.update(1)

    print(f"\n🎵 All MIDI files converted! Total events processed: {total_events}")

# === Run batch processing ===
process_all_midis(INPUT_FOLDER, OUTPUT_FOLDER)
