import os
import sys
import time
import signal
import shutil
import logging
import subprocess

import soundfile as sf
import pretty_midi
from tqdm import tqdm

child_proc = None
terminate = False

# Setup logger to both terminal and file
log_file = "conversion_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()


def signal_handler(sig, frame):
    global child_proc
    global terminate
    logger.info("SIGINT received. Cleaning up and exiting...")

    if child_proc and child_proc.poll() is None:
        logger.info("Terminating running subprocess...")
        child_proc.kill()

    logger.info("Exiting...")
    terminate = True


def get_output_name(filename):
    return filename.rsplit('.', 1)[0] + '.wav'


def is_midi_clean(midi_path, min_velocity=10):
    ALLOWED_PIANO_PROGRAMS = set(range(0, 6))  # 0 to 5 = all piano types

    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        logger.warning(f"Failed to load MIDI: {midi_path} with error: {e}")
        return False

    for inst in midi.instruments:
        if inst.is_drum:
            logger.info(f"{midi_path} skipped: contains drum track")
            return False
        if inst.program not in ALLOWED_PIANO_PROGRAMS:
            name = pretty_midi.program_to_instrument_name(inst.program)
            logger.info(f"{midi_path} skipped: contains non-piano instrument ({name})")
            return False
        for note in inst.notes:
            if note.velocity < min_velocity:
                logger.info(f"{midi_path} skipped: contains low-velocity note (v={note.velocity})")
                return False

    return True



def midi_to_wav(midi_path, output_path, soundfont):
    global child_proc
    arg_list = [
        "fluidsynth",
        "-ni",
        "-F", output_path,
        "-r", "16000",
        soundfont,
        midi_path
    ]

    try:
        child_proc = subprocess.Popen(
            arg_list,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
        child_proc.wait()
        return child_proc.returncode
    except Exception as e:
        logger.error(f"Unexpected error during FluidSynth conversion: {e}")
        return 1


def force_mono(output_path):
    global child_proc
    temp_path = output_path.rsplit('.', 1)[0] + ".temp.wav"

    arg_list = [
        "ffmpeg",
        "-y",
        "-i", output_path,
        "-ac", "1",
        temp_path
    ]

    try:
        child_proc = subprocess.Popen(
            arg_list,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
        child_proc.wait()

        if child_proc.returncode == 0:
            os.replace(temp_path, output_path)
            return True
        else:
            logger.warning(f"FFmpeg conversion failed for: {output_path}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    except Exception as e:
        logger.error(f"Unexpected error during FFmpeg conversion: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def py_midi_to_wav(midi_file, output_path, soundfont):
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    if soundfont:
        audio_data = midi_data.fluidsynth(fs=16000, sf2_path=soundfont)
    else:
        audio_data = midi_data.fluidsynth(fs=16000)

    sf.write(output_path, audio_data, samplerate=16000)
    logger.info(f"Conversion complete: {output_path}")


def py_process_dir(input_dir, output_dir, completed_dir, baddies_dir, soundfont):
    global terminate
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(completed_dir, exist_ok=True)
    os.makedirs(baddies_dir, exist_ok=True)

    midi_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mid')]
    total_files = len(midi_files)
    processed_count = 0
    error_count = 0
    start_time = time.time()

    with tqdm(total=total_files, desc="Processing MIDI", unit="file") as pbar:
        for filename in midi_files:
            if terminate:
                break

            file_path = os.path.join(input_dir, filename)
            completed_path = os.path.join(completed_dir, filename)
            output_path = os.path.join(output_dir, get_output_name(filename))

            if is_midi_clean(file_path):
                try:
                    py_midi_to_wav(file_path, output_path, soundfont)
                    shutil.move(file_path, completed_path)
                    logger.info(f"converted and moved: {filename}")
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    error_count += 1
                    shutil.move(file_path, os.path.join(baddies_dir, filename))
            else:
                logger.info(f"{filename} failed MIDI checks. Moving to baddies.")
                shutil.move(file_path, os.path.join(baddies_dir, filename))
                error_count += 1


            pbar.update(1)

    elapsed = time.time() - start_time
    if terminate:
        logger.info(f"Processing interrupted. {processed_count}/{total_files} files processed.")
    else:
        logger.info(f"Processing complete. {processed_count}/{total_files} files processed in {elapsed:.2f}s.")


def process_dir(input_dir, output_dir, completed_dir, baddies_dir, soundfont):
    global terminate
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(completed_dir, exist_ok=True)
    os.makedirs(baddies_dir, exist_ok=True)

    midi_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mid')]
    total_files = len(midi_files)
    processed_count = 0
    error_count = 0
    start_time = time.time()

    with tqdm(total=total_files, desc="FluidSynth Processing", unit="file") as pbar:
        for filename in midi_files:
            if terminate:
                break

            file_path = os.path.join(input_dir, filename)
            completed_path = os.path.join(completed_dir, filename)
            output_path = os.path.join(output_dir, get_output_name(filename))

            return_code = midi_to_wav(file_path, output_path, soundfont)

            if return_code == 0 and force_mono(output_path):
                logger.info(f"converted and moved: {filename}")
                processed_count += 1
                shutil.move(file_path, completed_path)
            else:
                logger.warning(f"Conversion of {filename} failed with code {return_code}")
                error_count += 1
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.move(file_path, os.path.join(baddies_dir, filename))

            pbar.update(1)

    elapsed = time.time() - start_time
    if terminate:
        logger.info(f"Processing Interrupted. {processed_count}/{total_files} files processed.")
        sys.exit(1)
    else:
        logger.info(f"Processing complete. {processed_count}/{total_files} files processed in {elapsed:.2f}s.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    base = "./"
    input_directory = os.path.join(base, "hz_midi")
    output_directory = os.path.join(base, "hz_16k_wav")
    completed_directory = os.path.join(base, "hz_processed_midi")
    baddies_directory = os.path.join(base, "baddies")

    soundfont = "Timbres of Heaven (XGM) 4.00(G).sf2"

    py_process_dir(input_directory, output_directory, completed_directory, baddies_directory, soundfont)
