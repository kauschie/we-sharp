import os
import sys
import shutil
import logging
from datetime import datetime
import pandas as pd
import subprocess
import signal
import re
from boxsdk import Client, JWTAuth
import box_functions

class PrintLogger:
    def write(self, message):
        # print(message, end='')  # Always print the message exactly as received
        if message.strip():  # Log only non-empty lines
            logging.debug(message.strip())
    
    def flush(self):  # Handle the flush method for compatibility
        pass

class ErrorLogger:
    def write(self, message):
        # print(message, end='', file=sys.__stderr__)  # Print to stderr exactly as received
        if message.strip():  # Log only non-empty lines
            logging.error(message.strip())
    
    def flush(self):
        pass

# Define the SIGINT, SIGTERM, and SIGHUP handler
# use signal handler to handle early termination in the
#   event that i need to stop early or of a crash or something
df = None
def handle_exit_signal(signal_received, frame):
    logging.warning(f"Received signal {signal_received}. Terminating early and saving current DataFrame.")
    if df is not None:
        save_lookup_table(df)
    logging.info("DataFrame saved. Exiting program.")
    exit(0)
signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)
signal.signal(signal.SIGHUP, handle_exit_signal)


# Define paths
lookup_table_path = 'song_list.csv'
bak_dir = './bak'
# orig_dir = './orig'
orig_dir = '/mnt/c/Users/mkaus/Downloads/AppleMusicDecrypt-Windows_latest/downloads/orig'
# delete_dir = './delete'
delete_dir = '/mnt/c/Users/mkaus/Downloads/AppleMusicDecrypt-Windows_latest/downloads/delete'


# Box setup
auth = JWTAuth.from_settings_file('./keypair.json')
client = Client(auth)
# we_sharp_id = '284827830368' # we-sharp project root directory
song_list_root = '293564308799' # 90s folder currently
# music_dir = '288133514348'   # parent dir of 90s song dir
orig_remote_dir = '292504599665'   # sub dir of 90's orig music dir

# Ensure required directories exist
for directory in [bak_dir, orig_dir, delete_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize logging
logging.basicConfig(filename='pipeline.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting the music data pipeline.")
logging.getLogger("boxsdk").setLevel(logging.WARNING)
# Redirect stdout and stderr
# sys.stdout = PrintLogger()
sys.stderr = ErrorLogger()

# Function to create a backup of song_list.csv
def backup_lookup_table():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(bak_dir, f"song_list_{timestamp}.csv")
    shutil.copy(lookup_table_path, backup_path)
    logging.info(f"Backup created: {backup_path}")

def load_lookup_table():
    global df

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
    # artist, title = get_artist_title(filename)

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
    
    artist_match = re.search(r"\s*album_artist\s*:\s*(.+)", output)
    title_match = re.search(r"\s*title\s*:\s*(.+)", output)

    album_artist = artist_match.group(1) if artist_match else None
    title = title_match.group(1) if title_match else None

    logging.info(f"Extracted metadata for {filename}: Artist={album_artist}, Title={title}, Length={song_length}s")
    return album_artist, title, song_length

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
    # directory = 'music' # local
    directory = '/mnt/c/Users/mkaus/Downloads/AppleMusicDecrypt-Windows_latest/downloads/ready_to_preprocess'
    df = load_lookup_table()
    files = os.listdir(directory)

    for file in files:
        if file.endswith('.m4a'):
            filename = file
            artist, title, songLength = extract_metadata(os.path.join(directory, file))
            lrc_filename = filename.replace(".m4a", ".lrc")
            lrc_box_file_id = None
            
            if os.path.exists(os.path.join(directory, lrc_filename)):
                lrc_box_file_id = 'pending'

            df = add_song_to_lookup(df, filename, artist, title, songLength, lrc_box_file_id)
            
            shutil.move(os.path.join(directory, file), os.path.join(orig_dir, file))
            logging.info(f"Moved {file} to {orig_dir}.")
                
            if lrc_box_file_id == 'pending':
                shutil.move(os.path.join(directory, lrc_filename), os.path.join(orig_dir, lrc_filename))
                logging.info(f"Moved {lrc_filename} to {orig_dir}.")

    save_lookup_table(df)

# Function to update upload status in the DataFrame
def update_upload_status(df, filename, box_file_id, lrc_box_file_id=None):
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[(df['filename'] == filename), ['upload_status', 'upload_time', 'box_file_id', 'lrc_box_file_id']] = \
        ['uploaded', str(upload_time), str(box_file_id), str(lrc_box_file_id)]
    
    logging.info(f"Updated upload status for {filename}: box_file_id={box_file_id}, lrc_box_file_id={lrc_box_file_id}")
    return df

# Stub for uploading files and tracking with Box file ID
def upload_and_track_files():
    df = load_lookup_table()
    for filename in os.listdir(orig_dir):
        if filename.endswith('.m4a'):
            lrc_filename = filename.replace(".m4a", ".lrc")
            lrc_box_file_id = None

            # Call function from box_function.py (Uploads file to box)
            box_file_id = box_functions.upload_to_box(orig_dir, filename, orig_remote_dir, client)

            # Log result
            logging.info(f"{filename} was uploaded to Box with file_id: {box_file_id}.")

            if os.path.exists(os.path.join(orig_dir, lrc_filename)):
                lrc_box_file_id = box_functions.upload_to_box(orig_dir, lrc_filename, orig_remote_dir, client)
                logging.info(f"{lrc_filename} was uploaded to Box with file_id: {lrc_box_file_id}.")

            df = update_upload_status(df, filename, str(box_file_id), str(lrc_box_file_id))

            shutil.move(os.path.join(orig_dir, filename), os.path.join(delete_dir, filename))
            logging.info(f"Moved {filename} to {delete_dir} after upload.")
            
            if lrc_box_file_id:
                shutil.move(os.path.join(orig_dir, lrc_filename), os.path.join(delete_dir, lrc_filename))
                logging.info(f"Moved {lrc_filename} to {delete_dir} after upload.")

    save_lookup_table(df)
    logging.info("Completed upload and tracking for all files in orig.")

def upload_lrcs(directory = None):
    # directory = 'music' # local
    non_matches = []
    if directory == None:
        directory = '/mnt/c/Users/mkaus/Downloads/AppleMusicDecrypt-Windows_latest/downloads/ready_to_preprocess'
    df = load_lookup_table()
    files = os.listdir(directory)
    hasChanged = False

    for file in files:
        if file.endswith('.lrc'):
            id = file.replace(".lrc", ".m4a")
            if not (df['filename'] == id).any():
                non_matches.append((id, file))
                # logging.info(f"Warning: No match for filename '{id}' for associated lrc file {file}")
            else:
                # upload
                lrc_box_file_id = box_functions.upload_to_box(directory, file, orig_remote_dir, client)
                if not lrc_box_file_id:
                    logging.error(f"Failed to upload {file}. Skipping...")
                    continue

                # update file id
                df.loc[df['filename'] == id, 'lrc_box_file_id'] = lrc_box_file_id
                logging.info(f"Updated lrc_box_file_id for '{id}' with {lrc_box_file_id}")
                
                # mark that changes have been made
                hasChanged = True
                
                # move file to be deleted
                shutil.move(os.path.join(directory, file), os.path.join(delete_dir, file))
                logging.info(f"Moved {file} to {delete_dir} after upload.")

    if non_matches:
        logging.warning(f"{len(non_matches)} unmatched (m4a,lrc) pairs: {non_matches}")

    if hasChanged:
        save_lookup_table(df)
        upload_song_list()
    else:
        logging.info(f"Did not update any files.")

# Fixes missing file IDs in song_list.csv
#   if they incorrectly say 'pending'
def fix_ids():
    df = load_lookup_table()

    if df.empty:
        logging.error(f"song_list.csv not found at {lookup_table_path}")
        return

    # Retrieve all items in the specified Box folder
    folder_id = orig_remote_dir
    items = box_functions.get_all_items(client, folder_id)
    unique_items = list({item.id: item for item in items}.values())
    logging.info(f"Got {len(items)} items from Box...")
    logging.info(f"There's {len(unique_items)} unique items")

    # Create a dictionary for easy lookup of Box file IDs by filename
    box_items = {}
    for item in unique_items:
        box_timestamp = item['created_at']
        dt = datetime.fromisoformat(box_timestamp)
        formatted_timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
        box_items[item.name] = ['uploaded', str(item.id), str(formatted_timestamp)]
        logging.info(f"Finished getting upload time from {item.id}")

    # Update dataframe
    updates_made = False
    for idx, row in df.iterrows():
        filename = row['filename']

        # test to see if lrc files need to be fixed
        if row['box_file_id'] == 'pending' or \
            row['upload_status'] == 'pending' or \
            row['lrc_box_file_id'] == 'pending':

            if filename in box_items:
                lrc_filename = filename.replace(".m4a", ".lrc")
                lrc_id = box_items.get(lrc_filename, 'pending') # Get 'pending' if not found
                if lrc_id != 'pending':
                    lrc_id = lrc_id[1] # guard against lrc file not found

                    # shouldn't need to update lrc if the lrc_id isn't found
                    new_vals = {
                        'upload_status': box_items[filename][0],
                        'box_file_id': box_items[filename][1],
                        'upload_time': box_items[filename][2],
                        'lrc_box_file_id': lrc_id  # Add lrc ID if needed in the DataFrame
                    }
                    
                    df.loc[idx, new_vals.keys()] = new_vals.values()
                    updates_made = True
                    logging.info(f"Updated {filename} with file ID: {box_items[filename]}")

        # test to see if id needs to be updated
        elif (filename in box_items) and (row['box_file_id'] != box_items[filename][1]):
            print(f"fix ids found mixmatching ids for {filename}")
            print(f"old_id: {row['box_file_id']} new_id: {box_items[filename][1]}")
            new_vals = {
                    'box_file_id': box_items[filename][1],  # new id
                    'upload_time': box_items[filename][2]   # upload_time in case its different
                }
            df.loc[idx, new_vals.keys()] = new_vals.values()
            updates_made = True
            logging.info(f"Updated {filename} with file ID: {box_items[filename]}")

    if updates_made:
        save_lookup_table(df)
        logging.info("Updated song_list.csv with missing file IDs. Uploading to Box")
        upload_song_list()
    else:
        upload_song_list()
        logging.info("No missing file IDs were found in song_list.csv.")


# upload to we-sharp/
def upload_song_list():
    box_functions.upload_to_box(".", lookup_table_path, song_list_root, client)
    logging.info("Finished uploading song_list.csv")


def main():
    process_files()
    upload_and_track_files()
    upload_song_list()

if __name__ == "__main__":
    main()
