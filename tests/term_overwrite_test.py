import time
import sys
import os

def log_event(filename, processed_count, total_files, start_time):
    elapsed_time = time.time() - start_time

    # Log the current event on a new line
    print(f"\r\033[KProcessed and moved: {filename}")

    # Move to the bottom line, clear it, and update the progress and timer
    print(f"Processed {processed_count}/{total_files} files... Elapsed Time: {elapsed_time:.2f}s", end="")
    sys.stdout.flush()

# Example function to simulate processing
def simulate_processing(total_files):
    start_time = time.time()
    processed_count = 0

    for i in range(total_files):
        filename = f"file_{i+1}.mid"
        processed_count += 1

        # Simulate processing delay
        time.sleep(0.1)

        # Log the event and update the progress line
        log_event(filename, processed_count, total_files, start_time)

# Example usage
if __name__ == "__main__":
    simulate_processing(1000)