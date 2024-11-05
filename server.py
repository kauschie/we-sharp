import pandas as pd
import random
import os
import shutil
import logging
from datetime import datetime
import argparse
from boxsdk import Client, JWTAuth
import box_functions

# Paths for files and directories
song_list_path = 'song_list.csv'
song_list_id = None
cut_list_path = 'cut_list.csv'
cut_config_path = 'cut_config.txt'
bak_dir = './bak'

# Box setup
auth = JWTAuth.from_settings_file('./keypair.json')
client = Client(auth)
# music folder id (Hardcode?)
box_root_folder_id = '288133514348'
# we-sharp folder id (Do we want to have this hardcoded?)
we_sharp_id = '284827830368'

# Initialize logging
logging.basicConfig(filename='server_pipeline.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the backup directory exists
# should be local to hold backups and sync them back later
if not os.path.exists(bak_dir):
    os.makedirs(bak_dir)

# Backup the cut_list.csv before processing
def backup_cut_list():
    if os.path.exists(cut_list_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(bak_dir, f"cut_list_{timestamp}.csv")
        shutil.copy(cut_list_path, backup_path)
        logging.info(f"Backup created for cut_list.csv: {backup_path}")

# Stub function to pull song_list.csv from Box
# Integrate with Dominic with BoxAPI
def get_song_list():

    logging.info("Fetching song_list.csv from Box.")
    # Use current directory
    directory = "."
    global song_list_path

    # Determine if the file path already exists, if yes delete
    file_path = os.path.join(directory, song_list_path)
    logging.info(f"Searching for {file_path}.")
    if os.path.exists(file_path):
        logging.info(f"{file_path} found. Deleting the file")
        os.remove(file_path)

    logging.info(f"Beginning to download csv from box. Will be located at {file_path}")
    # Download the csv from box. It should be in the folder 'we-sharp'
    song_list_path = box_functions.download_from_box(directory, song_list_path, we_sharp_id, client)

    logging.info(f"Pull from box complete. Returning: {song_list_path}")
    return song_list_path

# Stub function to sync files with Box
# Integrate with Dominic with BoxAPI
def sync_with_box():
    logging.info("Syncing cut_list.csv and backup directory with Box (stubbed).")

    # The location of cut_list.csv
    directory = "."

    # I assume this function takes cut_list.csv and uploads it to box
    # First we need to Delete the cut_list.csv file from box
    # If we do not want to delete the original file first, we could upload first, delete the old file, then rename the uploaded file

    # Search for cut_list.csv on box. If found then delete the file
    if box_functions.delete_from_box(cut_list_path, we_sharp_id, client) == False:
        logging.info("cut_list.csv was not deleted. It most likely did not exist.")
    else:
        logging.info("cut_list.csv was successfully deleted.")

    # Upload local version of cut_list.csv to box
    logging.info(f"Uploading {cut_list_path} to box.")
    cut_list_path_id = box_functions.upload_to_box(directory, cut_list_path, we_sharp_id, client)
    logging.info(f"{cut_list_path} is now synced with box. Box File ID: {cut_list_path_id}")

    # Now we need to sync the backup directory "./bak_dir" with box

# Function to read the cut config file
def read_cut_config():
    config = {}
    if os.path.exists(cut_config_path):
        with open(cut_config_path, 'r') as file:
            for line in file:
                key, value = line.strip().split('=')
                config[key] = int(value)
        logging.info(f"Read cut configuration from file: {config}")
    else:
        # If no config file, use defaults (can be overwritten by argparse)
        config = {'Start_offset': 30, 'length': 10, 'n_cuts': 3}
        logging.info(f"No config file found. Using default configuration: {config}")
    return config

# Function to generate cut times based on song duration and config
def generate_cut_times(duration, start_offset, length, n_cuts):
    if duration <= start_offset + length:
        logging.warning(f"Song duration ({duration}s) too short for requested cuts.")
        return []

    cut_times = []
    end_limit = int(duration - length)

    for _ in range(n_cuts):
        cut_start = random.randint(start_offset, end_limit)
        cut_times.append(cut_start)
    cut_times.sort()
    return cut_times

# Function to update file IDs in cut_list.csv after uploading
def update_cut_file_ids(file_name, drums_id, vocals_id, bass_id, other_id):
    if not os.path.exists(cut_list_path):
        logging.error("cut_list.csv not found. Cannot update file IDs.")
        return

    cut_df = pd.read_csv(cut_list_path)
    
    # Update the file IDs for the specified entry
    cut_df.loc[cut_df['file_name'] == file_name, ['drums_file_id', 'vocals_file_id', 'bass_file_id', 'other_file_id']] = \
        [drums_id, vocals_id, bass_id, other_id]

    cut_df.to_csv(cut_list_path, index=False)
    logging.info(f"Updated file IDs for {file_name} in cut_list.csv.")

def create_or_update_cut_list(start_offset, cut_length, n_cuts):
    # Backup the current cut_list.csv if it exists
    #get_cut_list() # TODO -> get from box first
    if os.path.exists(cut_list_path):
        backup_cut_list()

    # Pull the song_list.csv from Box
    get_song_list() # TODO: Implement with Dom

    # Load the song_list.csv
    if not os.path.exists(song_list_path):
        logging.error("song_list.csv not found. Exiting process.")
        return
    song_df = pd.read_csv(song_list_path)

    # Prepare the cut_list DataFrame or load existing with specified dtypes

    # FOR THE CUT LIST FUNCTION
    if os.path.exists(cut_list_path):
        cut_df = pd.read_csv(cut_list_path, dtype='object')
    else:
        # Initialize an empty DataFrame with compatible data types
        cut_df = pd.DataFrame(columns=[
            'file_name', 'file_id', 'song_length', 'process_time', 'start_offset',
            'cut_length', 'n_cuts', 'drums_file_id', 'vocals_file_id', 'bass_file_id',
            'other_file_id'] + [f'cut_time{i+1}' for i in range(n_cuts)
        ])
        cut_df = cut_df.astype('object')

    # Process each song in song_list.csv
    for _, row in song_df.iterrows():
        file_name = row['filename']
        file_id = row['box_file_id']
        song_length = row['songLength']
        
        # Generate cut times
        cut_times = generate_cut_times(song_length, start_offset, cut_length, n_cuts)
        if not cut_times:
            logging.info(f"Skipping {file_name} due to insufficient duration.")
            continue

        # Prepare entry for cut_list
        cut_entry = {
            'file_name': file_name,
            'file_id': str(file_id),
            'song_length': str(song_length),
            'process_time': 'pending',
            'start_offset': str(start_offset),
            'cut_length': str(cut_length),
            'n_cuts': str(n_cuts),
            'drums_file_id': 'pending',
            'vocals_file_id': 'pending',
            'bass_file_id': 'pending',
            'other_file_id': 'pending'
        }
        for i, cut_time in enumerate(cut_times):
            cut_entry[f'cut_time{i+1}'] = str(cut_time)

        # Check if the entry already exists in cut_df
        if file_name in cut_df['file_name'].values:
            # Get the index of the existing row
            row_index = cut_df.index[cut_df['file_name'] == file_name].tolist()[0]

            # Update each field in the existing row
            for key, value in cut_entry.items():
                # print(f"Updating '{key}' for '{file_name}': {value}")
                cut_df.at[row_index, key] = value  # Directly set the cell value
            logging.info(f"Updated entry for {file_name} in cut_list.csv.")
        else:
            # Add as a new row if it does not exist
            cut_df = pd.concat([cut_df, pd.DataFrame([cut_entry])], ignore_index=True)
            logging.info(f"Added new entry for {file_name} in cut_list.csv.")

    # Save the updated cut_list.csv
    cut_df.to_csv(cut_list_path, index=False)
    logging.info("cut_list.csv has been updated with new cut times.")

    # Sync with Box
    # TODO: Implement with Dominic
    sync_with_box()



# Stub function for preprocessing audio based on cut_list.csv
def preprocess_audio():
    logging.info("Starting audio preprocessing (stubbed).")
    # TODO: Implement preprocessing logic with Demucs and FFmpeg for each entry in cut_list.csv



# Main function with argument parsing
def main():
    parser = argparse.ArgumentParser(description="Manage and preprocess audio track cuts using cut_list.csv.")
    parser.add_argument(
        '--mode', choices=['gencuts', 'preprocess'], required=True,
        help="Operation mode: 'gencuts' to create/update cut_list, 'preprocess' to process audio based on cut_list."
    )
    parser.add_argument('--start_offset', type=int, default=None, help="Start offset for random cuts.")
    parser.add_argument('--length', type=int, default=None, help="Length of each cut.")
    parser.add_argument('--n_cuts', type=int, default=None, help="Number of cuts per audio file.")

    args = parser.parse_args()
    cut_config = read_cut_config()

    start_offset = args.start_offset if args.start_offset is not None else cut_config['Start_offset']
    cut_length = args.length if args.length is not None else cut_config['length']
    n_cuts = args.n_cuts if args.n_cuts is not None else cut_config['n_cuts']

    # Execute based on mode
    if args.mode == 'gencuts':
        create_or_update_cut_list(start_offset, cut_length, n_cuts)
    elif args.mode == 'preprocess':
        preprocess_audio()
if __name__ == "__main__":
    main()
