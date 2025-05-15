import pretty_midi
import argparse
import os

def inspect_midi(midi_path):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Failed to load {midi_path}: {e}")
        return

    print(f"\n🧾 Inspecting MIDI file: {midi_path}")
    print("-" * 60)

    for i, inst in enumerate(midi.instruments):
        prog = inst.program
        name = pretty_midi.program_to_instrument_name(prog)
        track_type = "Drums" if inst.is_drum else "Instrument"
        notes = len(inst.notes)

        print(f"Track {i}: {track_type}")
        print(f"  Program: {prog} - {name}")
        print(f"  Num Notes: {notes}")
        print(f"  Channel: {inst.program if not inst.is_drum else 9}")
        print()

    print("-" * 60)
    print(f"Total tracks: {len(midi.instruments)}")

def dump_note_stats(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    for i, inst in enumerate(midi.instruments):
        print(f"Track {i}: {pretty_midi.program_to_instrument_name(inst.program)}")
        for note in inst.notes:
            duration = note.end - note.start
            print(f"  Pitch: {note.pitch}, Velocity: {note.velocity}, Duration: {duration:.4f}s")

def fix_velocities(midi_path, output_path, min_velocity=50):
    midi = pretty_midi.PrettyMIDI(midi_path)

    for inst in midi.instruments:
        for note in inst.notes:
            if note.velocity <= 1:
                note.velocity = min_velocity  # Replace inaudible velocities
            else:
                note.velocity = max(note.velocity, min_velocity)

    midi.write(output_path)
    print(f"✅ Fixed MIDI written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect MIDI files.")
    parser.add_argument("midi_path", type=str, help="Path to the MIDI file to inspect.")
    args = parser.parse_args()

    midi_path = args.midi_path

    if not os.path.exists(midi_path):
        print(f"Error: The file {midi_path} does not exist.")
        exit(1)

    inspect_midi(midi_path)
    dump_note_stats(midi_path)
    output_path = midi_path.replace(".mid", "_fixed.mid")
    fix_velocities(midi_path, output_path)
    inspect_midi(output_path)
    dump_note_stats(output_path)
    print(f"Fixed MIDI file saved to {output_path}")
    