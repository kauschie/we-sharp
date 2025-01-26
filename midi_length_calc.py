import os
import math
import time
import sys
import csv
import shutil
import signal
from mido import MidiFile
from collections import Counter

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

def analyze_durations(csv_filename, max_duration_minutes=120):
    """
    Analyze the durations in the CSV and produce stats.
    Bins durations into [0,1), [1,2), ..., [max_duration_minutes-1, max_duration_minutes)
    and then lumps any durations >= max_duration_minutes into a single ">= max_duration_minutes" bin.
    Prints out the count in each bin plus a cumulative total.
    """
    durations = []

    # --- 1) Read CSV and collect durations ---
    with open(csv_filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header if present
        for row in reader:
            # row: [filename, filesize, duration]
            try:
                duration_seconds = float(row[2])
                # Filter out negative or obviously invalid durations:
                if duration_seconds < 0:
                    continue
                durations.append(duration_seconds)
            except (ValueError, IndexError):
                # If something's off in the row format, skip or handle appropriately
                continue

    if not durations:
        print("No valid duration data found.")
        return

    # --- 2) Basic statistics ---
    min_duration = min(durations)
    max_duration = max(durations)
    avg_duration = sum(durations) / len(durations)

    total_files = len(durations)
    print(f"Number of files: {total_files}")
    print(f"Minimum duration (seconds): {min_duration:.2f}")
    print(f"Maximum duration (seconds): {max_duration:.2f}")
    print(f"Average duration (seconds): {avg_duration:.2f}\n")

    # --- 3) Build a histogram using a Counter (dictionary) ---
    distribution = Counter()

    for d in durations:
        # Convert seconds to minutes and find bin
        bin_index = int(d // 60)  # integer division to get minute bin
        if bin_index >= max_duration_minutes:
            # Lump everything beyond the cutoff into a single bin
            distribution[max_duration_minutes] += 1
        else:
            distribution[bin_index] += 1

    # --- 4) Print out the distribution with a running cumulative total ---
    print("Distribution of durations in 1-minute bins (count, cumulative):")

    cumulative = 0
    for i in range(max_duration_minutes):
        count = distribution[i]
        cumulative += count
        print(f"[{i},{i+1}) minutes: {count} file(s) (cumulative: {cumulative}) (percentage: {cumulative/total_files})")

    # Print the lumped bin if there is any
    if distribution[max_duration_minutes] > 0:
        count = distribution[max_duration_minutes]
        cumulative += count
        print(f"[>= {max_duration_minutes} minutes]: {count} file(s) (cumulative: {cumulative}) (percentage: {cumulative/total_files})")

def move_big_files(csv_filename, base_dir, duration_cutoff=300):
    """
    Reads a CSV file with columns [filename, filesize, duration].
    For each file whose duration is greater than duration_cutoff (default = 5 minutes),
    move it from base_dir into a subfolder named 'big'.
    
    :param csv_filename: Path to the CSV file.
    :param base_dir: Base directory where the files currently reside.
    :param duration_cutoff: Duration in seconds above which files should be moved.
    """
    big_dir = os.path.join(base_dir, 'big')
    small_dir = os.path.join(base_dir, 'small')
    # Create the 'big' subdirectory if it doesn't exist
    os.makedirs(big_dir, exist_ok=True)
    os.makedirs(small_dir, exist_ok=True)
    
    moved_count = 0
    total_count = 0
    big_count = 0
    small_count = 0
    skip_count = 0
    good_count = 0
    extra_count = 0
    
    with open(csv_filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header if present
        for row in reader:
            total_count += 1

            # Each row is [filename, filesize, duration]
            # Some CSVs might have extra columns, so we do a quick sanity check
            if len(row) < 3:
                extra_count += 1
                continue
            
            filename = row[0]
            # Convert duration string to float
            try:
                duration = float(row[2])
            except ValueError:
                # If conversion fails, skip this row
                skip_count += 1
                continue
            
            # If duration > 5 minutes (300 seconds), move the file
            if duration > duration_cutoff:
                # Build full paths
                source_path = os.path.join(base_dir, filename)
                dest_path = os.path.join(big_dir, filename)
                
                # Check if file actually exists before attempting to move
                if os.path.isfile(source_path):
                    big_count += 1
                    try:
                        shutil.move(source_path, dest_path)
                        moved_count += 1
                        print(f"Moved: {filename} -> {dest_path}")
                    except Exception as e:
                        print(f"Could not move file {filename}: {e}")
                        skip_count += 1
                else:
                    print(f"File not found, skipping: {filename}")
                    skip_count += 1
            elif duration <= 0:
                # Build full paths
                source_path = os.path.join(base_dir, filename)
                dest_path = os.path.join(small_dir, filename)
                # Check if file actually exists before attempting to move
                if os.path.isfile(source_path):
                    try:
                        small_count += 1
                        shutil.move(source_path, dest_path)
                        moved_count += 1
                        print(f"Moved: {filename} -> {dest_path}")
                    except Exception as e:
                        print(f"Could not move file {filename}: {e}")
                        skip_count += 1
                else:
                    print(f"File not found, skipping: {filename}")
                    skip_count += 1
            else:
                good_count += 1
    
    print(f"\nProcessed {total_count} rows from CSV.")
    print(f"Skipped {skip_count} rows from CSV.")
    print(f"{good_count} Good files found.")
    print(f"Moved {big_count} files into '{big_dir}' (duration > {duration_cutoff} seconds).")
    print(f"Moved {small_count} files into '{small_dir}' (duration <= 0 seconds).")


if __name__ == "__main__":
    # Register signal handler for SIGINT
    signal.signal(signal.SIGINT, signal_handler)

    # Define input and output directories, and CSV file path

    # 1) Real:


    # 2) Testing:
    # base = "./"
    # input_directory = os.path.join(base, "midi")


    csv_filepath = "./midi_metadata.csv"

    # Step 1: analyze all midi files
    # Run the MIDI processing function
    # input_directory = "/mnt/c/Users/mkaus/Downloads/The_Magic_of_MIDI/MIDI"
    # output_directory = "/mnt/c/Users/mkaus/Downloads/The_Magic_of_MIDI/done/MIDI"
    # process_midi_files(input_directory, output_directory, csv_filepath)

    # Step 2: get distribution
    # Run stats on MIDI files
    # analyze_durations(csv_filepath)

    # step 3: partition large/too small files at 300s cutoff
    input_directory = "/mnt/c/Users/mkaus/Downloads/The_Magic_of_MIDI/done/MIDI"
    move_big_files(csv_filepath, input_directory, duration_cutoff=300)
