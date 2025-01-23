import pretty_midi
import soundfile as sf
import time
import sys
import signal
import os
import shutil

terminate = False

def signal_handler(sig, frame):
    """Handle SIGINT signal to allow graceful shutdown."""
    global terminate
    print("\nSIGINT received. Cleaning up and exiting...")
    terminate = True

def format_time(seconds):
    """Convert seconds into a user-friendly hours, minutes, seconds format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)

def clear_print(msg, ending="\n"):
    print(f"\r\033[K{msg}", end=ending)

def get_output_name(filename):
    return filename.rsplit('.', 1)[0] + '.wav'

prev_estimate = 0
def log_event(filename, processed_count, error_count, total_files, start_time):
    global prev_estimate
    elapsed_time = time.time() - start_time

    # calc estimated time remaining
    if processed_count % 5 == 0 or processed_count == total_files:
        if processed_count > 0:
            avg = elapsed_time / processed_count
            remaining = total_files - processed_count
            prev_estimate = avg * remaining
        else:
            prev_estimate = 0


    time_remaining = format_time(prev_estimate)

    # Log the current event on a new line
    clear_print(f"converted and moved: {filename}")

    # Move to the bottom line, clear it, and update the progress, timer, and estimated time remaining
    print(f"Processed {processed_count}/{total_files} files. {error_count} conversion errors. Elapsed Time: {elapsed_time:.2f}s | "
          f"Est. Time Remaining: {time_remaining}", end="")
    sys.stdout.flush()

def midi_to_wav(midi_file, output_path, soundfont):
    """
    Converts a MIDI file to WAV without real-time playback using PrettyMIDI.
    
    Args:
        midi_file (str): Path to the MIDI file.
        soundfont (str): Path to the SoundFont file (.sf2).
        output_path (str): Path to save the output WAV file.
    """

    # Load the MIDI file with PrettyMIDI
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    # Synthesize audio using the given SoundFont
    if soundfont is not None:
        audio_data = midi_data.fluidsynth(fs=22050, sf2_path=soundfont)
    else:
        audio_data = midi_data.fluidsynth(fs=22050)
        

    # Write the audio data to a WAV file
    sf.write(output_path, audio_data, samplerate=22050)
    clear_print(f"Conversion complete: {output_path}", ending="")

def process_dir(input_dir, output_dir, completed_dir, soundfont):
    """
    Process MIDI files from an input directory and saves as 1-ch 22kHz WAV files.

    Parameters:
        input_dir (str): Directory containing the MIDI files to process.
        output_dir (str): Directory to move processed files.
        soundfont (str): Path to the soundfont file (or None if using default).
    """
    # Ensure directories exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(completed_dir, exist_ok=True)

    try:

        # Get the total number of files to process
        midi_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mid')]
        total_files = len(midi_files)

        processed_count = 0     # record number processed
        error_count = 0         # record number of files with errors
        start_time = time.time()    # start timer

        for filename in midi_files:
            if terminate:
                break # exit loop if sigint received

            file_path = os.path.join(input_dir, filename)
            completed_path = os.path.join(completed_dir, filename)
            output_name = get_output_name(filename)
            output_path = os.path.join(output_dir, output_name) 

            try:
                midi_to_wav(file_path, output_path, soundfont) # saves wav to disk in right location
                shutil.move(file_path, completed_path) # moves midi file to completed dir

                # Update progress
                processed_count += 1
                log_event(filename, processed_count, error_count, total_files, start_time)

            except Exception as e:
                clear_print(f"Error processing {filename}: {e}", ending="")
                error_count += 1


        # Final message
        if terminate:
            clear_print(f"Processing interrupted. {processed_count}/{total_files} files processed.")
        else:
            clear_print(f"Processing complete. {processed_count}/{total_files} files processed.")


    except Exception as e:
        # Handle any unexpected errors
        clear_print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Register signal handler for SIGINT
    signal.signal(signal.SIGINT, signal_handler)

    # example file for iterative test
    midi_file = "Foreigner_-_Cold_as_ice.mid"
    soundfont = "Timbres of Heaven (XGM) 4.00(G).sf2"
    output_wav = "midi-pp-output.wav"

    # paths for the real shebang
    input_directory = "/mnt/c/Users/mkaus/Downloads/The_Magic_of_MIDI/done/MIDI"
    output_directory = "/mnt/c/Users/mkaus/Downloads/The_Magic_of_MIDI/processed_wav"
    completed_directory = "/mnt/c/Users/mkaus/Downloads/The_Magic_of_MIDI/processed_orig"

    # # test one file
    # midi_to_wav(midi_file, soundfont, output_wav)

    # process directory
    process_dir(input_directory, output_directory, completed_directory, soundfont)