# other dependencies
# fluidsynth

# imports
# import pretty_midi
import soundfile as sf
import pretty_midi
import time
import sys
import signal
import subprocess
import os
import shutil

child_proc = None
terminate = False

def clear_print(msg, ending="\n"):
    print(f"\r\033[K{msg}", end=ending)

def signal_handler(sig, frame):
    """Handle SIGINT signal to allow graceful shutdown."""
    global child_proc
    global terminate
    print("\nSIGINT received. Cleaning up and exiting...")
    
    # terminate child process if running
    if child_proc and child_proc.poll() is None: # check if running
        clear_print("Terminating running subprocess...")
        child_proc.kill()  # Forcefully kill if it doesn't terminate

    clear_print("Exiting...")
    terminate = True
    # os._exit(1)  # Ensure immediate termination of the script

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

    # Move to the bottom line, clear it, and update the progress, timer, and estimated time remaining
    clear_print(f"Processed {processed_count}/{total_files} files. {error_count} errors. \tElapsed Time: {elapsed_time:.2f}s | "
          f"Est. Time Remaining: {time_remaining}", ending="")
    sys.stdout.flush()

def midi_to_wav(midi_path, output_path, soundfont):
    """
    Converts a MIDI file to WAV using FluidSynth.

    Args:
        midi_path (str): Path to the MIDI file.
        output_path (str): Path to the output WAV file.
        soundfont (str): Path to the SoundFont file.

    Returns:
        int: Exit code of the FluidSynth process (0 for success, non-zero for failure).
    """
    global child_proc
    arg_list = [
        "fluidsynth",
        "-ni",
        "-F", output_path,       # Output WAV file
        "-r", "22050",           # Sampling rate: 22050 Hz
        soundfont,               # Path to SoundFont
        midi_path                # Input MIDI file
    ]

    try:
        child_proc = subprocess.Popen(
            arg_list,
            stdout=subprocess.DEVNULL,  # Suppress stdout
            stderr=subprocess.PIPE,     # Capture stderr for debugging
            text=True
        )
        child_proc.wait()  # Wait for the process to finish
        return child_proc.returncode
    except Exception as e:
        clear_print(f"Unexpected error during FluidSynth conversion: {e}")
        return 1  # Non-zero exit code for failure


def force_mono(output_path):
    """
    Converts a WAV file to mono using FFmpeg, overwriting the original file.

    Args:
        output_path (str): Path to the stereo WAV file.

    Returns:
        int: Exit code of the FFmpeg process (0 for success, non-zero for failure).
    """
    global child_proc
    temp_path = output_path.rsplit('.', 1)[0] + ".temp.wav"

    arg_list = [
        "ffmpeg",
        "-y",             # Overwrite output without prompting
        "-i", output_path,  # Input file
        "-ac", "1",         # Force mono audio
        temp_path         # Overwrite original file
    ]

    try:
        child_proc = subprocess.Popen(
            arg_list,
            stdout=subprocess.DEVNULL,  # Suppress stdout
            stderr=subprocess.PIPE,     # Capture stderr for debugging
            text=True
        )
        child_proc.wait()  # Wait for the process to finish
        # child_proc = subprocess.Popen(arg_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # stdout, stderr = child_proc.communicate()
        # print(f"stdout: {stdout}")
        # print(f"stderr: {stderr}")
        # print(f"return: {child_proc.returncode}")

        if child_proc.returncode == 0:
            os.replace(temp_path, output_path)
            # clear_print(f"Converted to mono and replaced: {output_path}", ending="")
            return True
        else:
            clear_print(f"FFmpeg conversion failed for: {output_path}")
            if os.path.exists(temp_path):
                os.remove(temp_path)  # Cleanup temporary file
            return False

    except Exception as e:
        clear_print(f"Unexpected error during FFmpeg conversion: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)  # Cleanup temporary file
        return False


## start

def py_midi_to_wav(midi_file, output_path, soundfont):
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

def py_process_dir(input_dir, output_dir, completed_dir, baddies_dir, soundfont):
    """
    Process MIDI files from an input directory and saves as 1-ch 22kHz WAV files.

    Parameters:
        input_dir (str): Directory containing the MIDI files to process.
        output_dir (str): Directory to move processed files.
        soundfont (str): Path to the soundfont file (or None if using default).
    """
    global terminate
    # Ensure directories exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(completed_dir, exist_ok=True)
    os.makedirs(baddies_dir, exist_ok=True)

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
                py_midi_to_wav(file_path, output_path, soundfont) # saves wav to disk in right location
                shutil.move(file_path, completed_path) # moves midi file to completed dir

                # Update progress
                clear_print(f"converted and moved: {filename}")
                processed_count += 1
                log_event(filename, processed_count, error_count, total_files, start_time)

            except Exception as e:
                clear_print(f"Error processing {filename}: {e}", ending="")
                error_count += 1
                shutil.move(file_path, os.path.join(baddies_dir, filename)) # moves midi file to completed dir


        # Final message
        if terminate:
            clear_print(f"Processing interrupted. {processed_count}/{total_files} files processed.")
        else:
            clear_print(f"Processing complete. {processed_count}/{total_files} files processed.")


    except Exception as e:
        # Handle any unexpected errors
        clear_print(f"An error occurred: {e}")


# End


def process_dir(input_dir, output_dir, completed_dir, baddies_dir, soundfont):
    """
    Process MIDI files from an input directory and saves as 1-ch 22kHz WAV files.

    Parameters:
        input_dir (str): Directory containing the MIDI files to process.
        output_dir (str): Directory to move processed files.
        soundfont (str): Path to the soundfont file (or None if using default).
    """
    global terminate
    # Ensure directories exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(completed_dir, exist_ok=True)
    os.makedirs(baddies_dir, exist_ok=True)

    # Get the total number of files to process
    midi_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mid')]
    total_files = len(midi_files)

    processed_count = 0     # record number processed
    error_count = 0         # record number of files with errors
    start_time = time.time()    # start timer

    for filename in midi_files:
        if terminate == True:
            break

        # clear_print(f"working on file {filename}", ending="")
        file_path = os.path.join(input_dir, filename)
        completed_path = os.path.join(completed_dir, filename)
        output_name = get_output_name(filename)
        output_path = os.path.join(output_dir, output_name) 


        return_code = midi_to_wav(file_path, output_path, soundfont) # saves wav to disk in right location


        if return_code == 0 and force_mono(output_path) == True:
            # Log the current event on a new line
            clear_print(f"converted and moved: {filename}")
            processed_count += 1
            shutil.move(file_path, completed_path) # moves midi file to completed dir
        else:
            clear_print(f"Conversion of {filename} failed with code {return_code}")
            error_count += 1
            if os.path.exists(output_path):
                os.remove(output_path)  # Cleanup temporary file
            shutil.move(file_path, os.path.join(baddies_dir, filename)) # moves midi file to completed dir

        log_event(filename, processed_count, error_count, total_files, start_time)



    # Final message
    if terminate == True:
        print(f"Processing Interrupted. {processed_count}/{total_files} files processed.")
        os.exit(1)
    else:
        print(f"Processing complete. {processed_count}/{total_files} files processed.")


if __name__ == "__main__":
    # Register signal handler for SIGINT
    signal.signal(signal.SIGINT, signal_handler)

    # example file for iterative test
    # midi_file = "Foreigner_-_Cold_as_ice.mid"
    soundfont = "Timbres of Heaven (XGM) 4.00(G).sf2"
    # output_wav = "midi-pp-output.wav"
    

    # paths for testing
    # base = "./"
    # input_directory = os.path.join(base, "midi")

    # paths for the real shebang
    base = "/mnt/c/Users/mkaus/Downloads/The_Magic_of_MIDI/"
    input_directory = os.path.join(base, "done/MIDI")



    output_directory = os.path.join(base, "processed_wav")
    completed_directory = os.path.join(base, "processed_orig")
    baddies_directory = os.path.join(base, "baddies")

    # # test one file
    # midi_to_wav(midi_file, soundfont, output_wav)

    # process directory
    # process_dir(input_directory, output_directory, completed_directory, baddies_directory, soundfont)
    py_process_dir(input_directory, output_directory, completed_directory, baddies_directory, soundfont)
