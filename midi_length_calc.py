import os
import time
import sys
import csv
import shutil
import signal
from mido import MidiFile

# Global flag to track interruptions
terminate = False
prev_estimate = 0

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


def log_event(filename, processed_count, total_files, start_time):
    global prev_estimate
    elapsed_time = time.time() - start_time

    # calc estimated time remaining
    if processed_count % 20 == 0 or processed_count == total_files:
        if processed_count > 0:
            avg = elapsed_time / processed_count
            remaining = total_files - processed_count
            prev_estimate = avg * remaining
        else:
            prev_estimate = 0


    time_remaining = format_time(prev_estimate)

    # Log the current event on a new line
    print(f"\r\033[KProcessed and moved: {filename}")

    # Move to the bottom line, clear it, and update the progress, timer, and estimated time remaining
    print(f"Processed {processed_count}/{total_files} files... Elapsed Time: {elapsed_time:.2f}s | "
          f"Estimated Time Remaining: {time_remaining}", end="")
    sys.stdout.flush()



def signal_handler(sig, frame):
    """Handle SIGINT signal to allow graceful shutdown."""
    global terminate
    print("\nSIGINT received. Cleaning up and exiting...")
    terminate = True

def process_midi_files(input_dir, output_dir, csv_file):
    """
    Process MIDI files from an input directory and log their details into a CSV file.

    Parameters:
        input_dir (str): Directory containing the MIDI files to process.
        output_dir (str): Directory to move processed files.
        csv_file (str): Path to the CSV file to append MIDI file details.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get the total number of files to process
        midi_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mid')]
        total_files = len(midi_files)
        processed_count = 0
        start_time = time.time()


        # Open or create the CSV file
        with open(csv_file, mode='a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'filesize', 'duration']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if the file is empty
            if os.stat(csv_file).st_size == 0:
                writer.writeheader()

            # Iterate over all .mid files in the input directory
            for filename in midi_files:
                if terminate:
                    break  # Exit loop if SIGINT is received

                file_path = os.path.join(input_dir, filename)

                try:
                    # Get file size
                    file_size = os.path.getsize(file_path)

                    # Calculate duration using mido
                    midi = MidiFile(file_path)
                    duration = sum(msg.time for msg in midi) if midi.type != 2 else 0  # Skip if type 2 (no timing)

                    # Append file details to CSV
                    writer.writerow({
                        'filename': filename,
                        'filesize': file_size,
                        'duration': duration
                    })

                    # Move the file to the output directory
                    shutil.move(file_path, os.path.join(output_dir, filename))

                    # Update progress
                    processed_count += 1
                    
                    # Log the event and update the progress line
                    log_event(filename, processed_count, total_files, start_time)

                except Exception as e:
                    print(f"\r\033[KError processing {filename}: {e}", end="")
                    total_files -= 1

        # Final message
        if terminate:
            print(f"\r\033[KProcessing interrupted. {processed_count}/{total_files} files processed.")
        else:
            print(f"\r\033[KProcessing complete. {processed_count}/{total_files} files processed.")

    except Exception as e:
        # Handle any unexpected errors
        print(f"An error occurred: {e}")
        print("Progress saved to CSV.")


if __name__ == "__main__":
    # Register signal handler for SIGINT
    signal.signal(signal.SIGINT, signal_handler)

    # Define input and output directories, and CSV file path
    input_directory = "/mnt/c/Users/mkaus/Downloads/The_Magic_of_MIDI/MIDI"
    output_directory = "/mnt/c/Users/mkaus/Downloads/The_Magic_of_MIDI/done/MIDI"
    csv_filepath = "./midi_metadata.csv"

    # Run the MIDI processing function
    process_midi_files(input_directory, output_directory, csv_filepath)
