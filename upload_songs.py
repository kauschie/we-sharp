import os
import shutil
import logging
from datetime import datetime
import pandas as pd
import subprocess
import re
import random

# Define paths
lookup_table_path = 'song_list.csv'
bak_dir = './bak'
orig_dir = './orig'
delete_dir = './delete'

# Initialize logging
logging.basicConfig(filename='pipeline.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting the music data pipeline.")

# Function to create a backup of song_list.csv
def backup_lookup_table():
    if not os.path.exists(bak_dir):
        os.makedirs(bak_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(bak_dir, f"song_list_{timestamp}.csv")
    shutil.copy(lookup_table_path, backup_path)
    logging.info(f"Backup created: {backup_path}")

# Load or initialize the lookup table as a DataFrame
def load_lookup_table():
    if os.path.exists(lookup_table_path):
        df = pd.read_csv(lookup_table_path)
        backup_lookup_table()  # Make a backup before modifying
    else:
        # Create a new DataFrame if file doesn't exist
        df = pd.DataFrame(columns=['filename', 'artist', 'title', 'songLength', 'upload_status', 'upload_time', 'box_file_id', 'lrc_box_file_id'])
        df.to_csv(lookup_table_path, index=False)
        logging.info("Created new song_list.csv with headers.")
    return df

# Save the lookup table DataFrame back to CSV
def save_lookup_table(df):
    df.to_csv(lookup_table_path, index=False)
    logging.info("Updated song_list.csv and saved changes.")

# Extract metadata using FFmpeg
def extract_metadata(filename):
    # Extract artist and title from filename (format: "Artist - Title.m4a")
    base_name = os.path.splitext(filename)[0]
    artist, title = base_name.split(" - ", 1)

    # Run FFmpeg to get metadata
    command = ["ffmpeg", "-i", filename]
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    output = result.stderr  # FFmpeg metadata usually goes to stderr

    # Extract song length with fractional seconds
    duration_match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d+)", output)
    if duration_match:
        hours = int(duration_match.group(1))
        minutes = int(duration_match.group(2))
        seconds = float(duration_match.group(3))
        song_length = hours * 3600 + minutes * 60 + seconds  # Total duration in seconds with fractions
    else:
        song_length = None  # If duration is not found, set to None

    logging.info(f"Extracted metadata for {filename}: Artist={artist}, Title={title}, Length={song_length}s")
    return artist, title, song_length, filename

# Add a song entry if it doesn't exist in the DataFrame
def add_song_to_lookup(df, filename, artist, title, songLength, lrc_box_file_id):
    # Check if song exists by filename
    if not (df['filename'] == filename).any():
        # Define new_entry as a DataFrame directly
        new_entry = pd.DataFrame([{
            'filename': filename,
            'artist': artist,
            'title': title,
            'songLength': songLength,
            'upload_status': 'pending',
            'upload_time': '',
            'box_file_id': '',
            'lrc_box_file_id': lrc_box_file_id
        }])
        
        # Concatenate the new entry as a DataFrame
        df = pd.concat([df, new_entry], ignore_index=True)
        logging.info(f"Added new entry to song_list.csv for {filename}.")
    else:
        logging.info(f"Entry already exists for {filename}. Skipping addition.")
    return df


# Process files in the current directory
def process_files():
    df = load_lookup_table()
    files = os.listdir('.')  # List files in the current directory

    for file in files:
        if file.endswith('.m4a'):
            artist, title, songLength, filename = extract_metadata(file)
            # Check for an accompanying .lrc file
            lrc_filename = f"{artist} - {title}.lrc"
            lrc_box_file_id = None  # Initialize as None
            
            if os.path.exists(lrc_filename):
                lrc_box_file_id = 'pending'  # Indicate that the lrc file will be processed

            # Add or update the song entry in the DataFrame
            df = add_song_to_lookup(df, filename, artist, title, songLength, lrc_box_file_id)
            
            # Move file to `orig` directory if it's already in the lookup
            shutil.move(file, os.path.join(orig_dir, file))
            logging.info(f"Moved {file} to {orig_dir}.")
                
            # Move the .lrc file to `orig` if it exists
            if lrc_box_file_id == 'pending':
                shutil.move(lrc_filename, os.path.join(orig_dir, lrc_filename))
                logging.info(f"Moved {lrc_filename} to {orig_dir}.")

    # Save the updated DataFrame
    save_lookup_table(df)


# Function to update upload status in the DataFrame
def update_upload_status(df, artist, title, box_file_id, lrc_box_file_id=None):
    # Current timestamp for upload time
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Locate the specific row by artist and title, then update the columns
    df.loc[(df['artist'] == artist) & (df['title'] == title), ['upload_status', 'upload_time', 'box_file_id', 'lrc_box_file_id']] = \
        ['uploaded', upload_time, box_file_id, lrc_box_file_id]
    
    logging.info(f"Updated upload status for {artist} - {title}: box_file_id={box_file_id}, lrc_box_file_id={lrc_box_file_id}")
    return df

def upload_to_box(file):
    file_id = random.randint(0, 9999999)
    logging.info(f"{file} was successfully uploaded to Box with file_id: {file_id}.")
    return file_id

# Stub for uploading files and tracking with Box file ID
def upload_and_track_files():
    df = load_lookup_table()
    for filename in os.listdir(orig_dir):
        if filename.endswith('.m4a'):
            artist, title, songLength, _ = extract_metadata(filename)

            # Check if the .lrc file is also present in `orig`
            lrc_filename = f"{artist} - {title}.lrc"
            lrc_box_file_id = None

            # Stub: Call Box API to upload file and get Box file ID
            box_file_id = upload_to_box(os.path.join(orig_dir, filename))
            if os.path.exists(os.path.join(orig_dir, lrc_filename)):
                lrc_box_file_id = upload_to_box(os.path.join(orig_dir, lrc_filename))

            # Update DataFrame with upload status and Box file ID
            df = update_upload_status(df, artist, title, box_file_id, lrc_box_file_id)

            # Move file and .lrc file to delete directory after upload
            shutil.move(os.path.join(orig_dir, filename), os.path.join(delete_dir, filename))
            logging.info(f"Moved {filename} to {delete_dir} after upload.")
            
            if lrc_box_file_id:
                shutil.move(os.path.join(orig_dir, lrc_filename), os.path.join(delete_dir, lrc_filename))
                logging.info(f"Moved {lrc_filename} to {delete_dir} after upload.")

    # Save the final state of the DataFrame
    save_lookup_table(df)
    logging.info("Completed upload and tracking for all files in orig.")

def main():
    process_files()
    upload_and_track_files()


if __name__ == "__main__":
    main()