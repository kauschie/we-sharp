# python midi_convert.py generated_100_768.mid test_mid2.mp3
# or import it as a module and use it like this:
# from midi_convert import py_midi_to_audio
# py_midi_to_audio(midi_file, output_path, soundfont="my_soundfont.sf2", sample_rate=24000)

import os
import torch
import torchaudio
import pretty_midi
import argparse

def py_midi_to_audio(midi_file, output_path, soundfont=None, sample_rate=24000):
    """
    Converts a MIDI file to WAV or MP3 using PrettyMIDI, FluidSynth, and torchaudio.
    
    Args:
        midi_file (str): Path to the MIDI file.
        output_path (str): Output file path (.wav or .mp3).
        soundfont (str, optional): Path to SoundFont (.sf2). If None, system default is used.
        sample_rate (int): Sample rate of the output audio (default: 24000Hz).
    """
    if not os.path.isfile(midi_file):
        raise FileNotFoundError(f"MIDI file not found: {midi_file}")
    if soundfont and not os.path.isfile(soundfont):
        raise FileNotFoundError(f"SoundFont file not found: {soundfont}")
    
    # Determine output format
    ext = os.path.splitext(output_path)[1].lower()
    if ext not in [".wav", ".mp3"]:
        raise ValueError("Output file extension must be .wav or .mp3")

    # Load MIDI and synthesize audio
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    audio_np = midi_data.fluidsynth(fs=sample_rate, sf2_path=soundfont)  # (n_samples,)

    # Convert to mono torch tensor: shape (1, n_samples)
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

    # Convert float32 in range [-1.0, 1.0] to int16
    audio_tensor = (audio_tensor * 32767.0).clamp(-32768, 32767).to(torch.int16)

    # Save using torchaudio in PCM format (int16)
    torchaudio.save(output_path, audio_tensor, sample_rate, format=ext[1:])

    print(f"[✓] Converted {midi_file} → {output_path} ({sample_rate}Hz)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a MIDI file to WAV or MP3 using PrettyMIDI and torchaudio.")
    parser.add_argument("midi_file", type=str, help="Path to the input MIDI file")
    parser.add_argument("output_path", type=str, help="Path for the output audio file (.wav or .mp3)")
    parser.add_argument("--soundfont", type=str, default=None, help="Optional path to SoundFont (.sf2) file")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Sample rate for the output audio (default: 24000Hz)")

    args = parser.parse_args()

    py_midi_to_audio(args.midi_file, args.output_path, soundfont=args.soundfont, sample_rate=args.sample_rate)
