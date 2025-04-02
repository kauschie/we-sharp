import os
import torch
import torchaudio
import pretty_midi

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

    # Save audio using torchaudio
    torchaudio.save(output_path, audio_tensor, sample_rate, format=ext[1:])
    print(f"[✓] Converted {midi_file} → {output_path} ({sample_rate}Hz)")
