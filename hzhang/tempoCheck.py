import importlib
import numpy as np
import BeatNet  # make sure load BeatNet
importlib.reload(BeatNet)  # reload BeatNet
import os
import csv
from BeatNet.BeatNet import BeatNet  # make sure BeatNet install correct

#===================================================================
music_folder = "processed_wav"  # Set Music folder name+++++++++++++++\\
#===================================================================


# Set BeatNet
estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)

# Define music folder and output folder
tempo_output_folder = os.path.join(music_folder, "tempo_output")

# Create output folder if it doesn't exist
os.makedirs(tempo_output_folder, exist_ok=True)

# Get all wav files from music folder
wav_files = [f for f in os.listdir(music_folder) if f.endswith('.wav')]

# Set numpy print options
np.set_printoptions(suppress=True, precision=2)

# Loop through all music files
for wav_file in wav_files:
    # Get full file path
    file_path = os.path.join(music_folder, wav_file)

    try:
        # Process WAV file with BeatNet
        output = estimator.process(file_path)
    except RuntimeError as e:
        print(f"Error processing {wav_file}: {e}")
        continue

    # Construct CSV file name and path
    csv_filename = os.path.splitext(wav_file)[0] + ".csv"
    csv_path = os.path.join(tempo_output_folder, csv_filename)

    # Save results to CSV file
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Time", "Beat"])  # Write header

        for item in output:
            csv_writer.writerow([f"{item[0]:.2f}", int(item[1])])

    print(f"Processed {wav_file} and saved to {csv_path}")

# Ensure safe model loading

