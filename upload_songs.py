import os
import shutil
import logging
from datetime import datetime
import pandas as pd
import subprocess
import re
from boxsdk import Client, JWTAuth
import box_functions

# Define paths
lookup_table_path = 'song_list.csv'
bak_dir = './bak'
orig_dir = './orig'
delete_dir = './delete'

# Box setup
auth = JWTAuth.from_settings_file('./keypair.json')
client = Client(auth)
music_dir = '288133514348'   # parent dir of orig_dir
orig_remote_dir = '292504599665'   # sub dir of music_dir
we_sharp_id = '284827830368'
# Ensure required directories exist
for directory in [bak_dir, orig_dir, delete_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize logging
logging.basicConfig(filename='pipeline.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting the music data pipeline.")

# Function to create a backup of song_list.csv
def backup_lookup_table():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(bak_dir, f"song_list_{timestamp}.csv")
    shutil.copy(lookup_table_path, backup_path)
    logging.info(f"Backup created: {backup_path}")

def load_lookup_table():
    if os.path.exists(lookup_table_path):
        backup_lookup_table()  # Make a backup before modifying
        # Load the CSV, ensuring types for existing entries
        df = pd.read_csv(
            lookup_table_path,
            dtype={
                'filename': 'object',
                'artist': 'object',
                'title': 'object',
                'songLength': 'float64',
                'upload_status': 'object',
                'upload_time': 'object',
                'box_file_id': 'object',
                'lrc_box_file_id': 'object'
            }
        )
    else:
        # Initialize DataFrame with specific column types
        df = pd.DataFrame({
            'filename': pd.Series(dtype='object'),
            'artist': pd.Series(dtype='object'),
            'title': pd.Series(dtype='object'),
            'songLength': pd.Series(dtype='float64'),
            'upload_status': pd.Series(dtype='object'),
            'upload_time': pd.Series(dtype='object'),
            'box_file_id': pd.Series(dtype='object'),
            'lrc_box_file_id': pd.Series(dtype='object')
        })
        df.to_csv(lookup_table_path, index=False)
        logging.info("Created new song_list.csv with headers.")
    return df

# Save the lookup table DataFrame back to CSV
def save_lookup_table(df):
    df.to_csv(lookup_table_path, index=False)
    logging.info("Updated song_list.csv and saved changes.")

def get_artist_title(filename):
    base_name = os.path.splitext(filename)[0]
    artist, title = base_name.split(" - ", 1)
    return artist, title

# Extract metadata using FFmpeg
def extract_metadata(filename):
    artist, title = get_artist_title(filename)

    command = ["ffmpeg", "-i", filename]
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    output = result.stderr

    duration_match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d+)", output)
    if duration_match:
        hours = int(duration_match.group(1))
        minutes = int(duration_match.group(2))
        seconds = float(duration_match.group(3))
        song_length = hours * 3600 + minutes * 60 + seconds
    else:
        song_length = None

    logging.info(f"Extracted metadata for {filename}: Artist={artist}, Title={title}, Length={song_length}s")
    return artist, title, song_length, filename

def add_song_to_lookup(df, filename, artist, title, songLength, lrc_box_file_id):
    if not (df['filename'] == filename).any():
        new_entry = pd.DataFrame([{
            'filename': filename,
            'artist': artist,
            'title': title,
            'songLength': songLength if pd.notna(songLength) else 0.0,
            'upload_status': 'pending',
            'upload_time': 'pending',
            'box_file_id': 'pending',
            'lrc_box_file_id': lrc_box_file_id if pd.notna(lrc_box_file_id) else 'pending'
        }])
        
        if not new_entry.isna().all(axis=None):
            df = pd.concat([df, new_entry], ignore_index=True)
            logging.info(f"Added new entry to song_list.csv for {filename}.")
        else:
            logging.info(f"Skipping empty entry for {filename}.")
    else:
        logging.info(f"Entry already exists for {filename}. Skipping addition.")
    return df

# Process files in the current directory
def process_files():
    df = load_lookup_table()
    files = os.listdir('.')

    for file in files:
        if file.endswith('.m4a'):
            artist, title, songLength, filename = extract_metadata(file)
            lrc_filename = f"{artist} - {title}.lrc"
            lrc_box_file_id = None
            
            if os.path.exists(lrc_filename):
                lrc_box_file_id = 'pending'

            df = add_song_to_lookup(df, filename, artist, title, songLength, lrc_box_file_id)
            
            shutil.move(file, os.path.join(orig_dir, file))
            logging.info(f"Moved {file} to {orig_dir}.")
                
            if lrc_box_file_id == 'pending':
                shutil.move(lrc_filename, os.path.join(orig_dir, lrc_filename))
                logging.info(f"Moved {lrc_filename} to {orig_dir}.")

    save_lookup_table(df)

# Function to update upload status in the DataFrame
def update_upload_status(df, artist, title, box_file_id, lrc_box_file_id=None):
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[(df['artist'] == artist) & (df['title'] == title), ['upload_status', 'upload_time', 'box_file_id', 'lrc_box_file_id']] = \
        ['uploaded', str(upload_time), str(box_file_id), str(lrc_box_file_id)]
    
    logging.info(f"Updated upload status for {artist} - {title}: box_file_id={box_file_id}, lrc_box_file_id={lrc_box_file_id}")
    return df

# Stub for uploading files and tracking with Box file ID
def upload_and_track_files():
    df = load_lookup_table()
    for filename in os.listdir(orig_dir):
        if filename.endswith('.m4a'):
            artist, title = get_artist_title(filename)
            lrc_filename = f"{artist} - {title}.lrc"
            lrc_box_file_id = None

            # Call function from box_function.py (Uploads file to box)
            box_file_id = box_functions.upload_to_box(orig_dir, filename, orig_remote_dir, client)
            # Log result
            logging.info(f"{filename} was uploaded to Box with file_id: {box_file_id}.")

            if os.path.exists(os.path.join(orig_dir, lrc_filename)):
                lrc_box_file_id = box_functions.upload_to_box(orig_dir, lrc_filename, orig_remote_dir, client)
                logging.info(f"{lrc_filename} was uploaded to Box with file_id: {lrc_box_file_id}.")

            df = update_upload_status(df, artist, title, str(box_file_id), str(lrc_box_file_id))

            shutil.move(os.path.join(orig_dir, filename), os.path.join(delete_dir, filename))
            logging.info(f"Moved {filename} to {delete_dir} after upload.")
            
            if lrc_box_file_id:
                shutil.move(os.path.join(orig_dir, lrc_filename), os.path.join(delete_dir, lrc_filename))
                logging.info(f"Moved {lrc_filename} to {delete_dir} after upload.")

    save_lookup_table(df)
    logging.info("Completed upload and tracking for all files in orig.")

# upload to we-sharp/
def upload_song_list():
    box_functions.upload_to_box(".", lookup_table_path, we_sharp_id, client)
    logging.info("Finished uploading song_list.csv")


def main():
    process_files()
    upload_and_track_files()
    upload_song_list() #TODO: upload song_list.csv

if __name__ == "__main__":
    main()
