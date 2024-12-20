import pandas as pd
import random
import os
import sys
import shutil
import logging
from datetime import datetime
import argparse
import subprocess
from boxsdk import Client, JWTAuth
import box_functions
import signal

# Initialize logging
logging.basicConfig(
    filename='server_pipeline.log',
    level=logging.DEBUG,  # Use DEBUG during development to capture all events
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger("boxsdk").setLevel(logging.WARNING)

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

sys.stdout = PrintLogger()
sys.stderr = ErrorLogger()

df = None
def handle_exit_signal(signal_received, frame):
    logging.warning(f"Received signal {signal_received}. Terminating early and saving current DataFrame.")
    if df is not None:
        try:
            save_lookup_table(df, cut_list_path)
        except Exception as e:
            logging.error(f"Failed to save DataFrame on termination: {e}")
        logging.info("DataFrame saved. Exiting program.")
        sys.exit(0)
        
        
    logging.info("DataFrame saved. Exiting program.")
    exit(0)

signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)
# signal.signal(signal.SIGHUP, handle_exit_signal)

# Paths for files and directories
song_list_path = 'song_list.csv'
song_list_id = '1699222338484'
cut_list_path = 'cut_list.csv'
cut_config_path = 'cut_config.txt'
bak_dir = './bak'

# Box setup
auth = JWTAuth.from_settings_file('./keypair.json')
client = Client(auth)

# Folder IDs (shouldn't change)
# we-sharp folder id
# we_sharp_id = '284827830368'
song_list_root = '293564308799' # 90s folder
# Bak folder id
bak_folder_id = '292534511993'
# music folder id
box_root_folder_id = '288133514348'

# just garbage values right now TODO: Replace with real values
dbo_folder_id = '292547314909'
orig_folder_id = '292504599665'
drums_folder_id = '292548609290'
vocals_folder_id = '292546617353'
bass_folder_id = '292547961183'
other_folder_id = '292548127080'
training_data_folder_id = '292623465429' # training1
training_cuts_dbo_id = '292621472410' # contains all cut/preprocessed files
training_cuts_other_id = '292620621147'



# Ensure the backup directory exists
# should be local to hold backups and sync them back later
if not os.path.exists(bak_dir):
    os.makedirs(bak_dir)

# Backup the cut_list.csv before processing
def make_backup(file_path):
    if os.path.exists(file_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(bak_dir, f"cut_list_{timestamp}.csv")
        shutil.copy(file_path, backup_path)
        logging.info(f"Backup created for {file_path}: {backup_path}")

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
        logging.info(f"{file_path} found.")
        make_backup(file_path)
        logging.info(f"Deleting {file_path}")
        os.remove(file_path)

    logging.info(f"Beginning to download csv from box. Will be located at {file_path}")
    # Download the csv from box. It should be in the folder 'we-sharp'
    song_list_path = box_functions.download_from_box(directory, song_list_path, song_list_id, client)

    logging.info(f"Pull from box complete.")

    if not os.path.exists(song_list_path):
        logging.error("song_list.csv not found. Exiting process.")
        return

    return pd.read_csv(song_list_path)


def get_cut_df(n_cuts=None):
    
    # FOR THE CUT LIST FUNCTION
    if os.path.exists(cut_list_path):
        make_backup(cut_list_path)   # backup if it already exists
        cut_df = pd.read_csv(cut_list_path, dtype='object')
        if n_cuts == None:  # just retrieving, not dropping cols or creating new cuts
            return cut_df

    else:
        # Initialize an empty DataFrame with compatible data types
        cut_df = pd.DataFrame(columns=[
            'file_name', 'file_id', 'song_length', 'process_time', 'start_offset',
            'cut_length', 'n_cuts', 'dbo_file_id', 'drums_file_id', 'vocals_file_id', 'bass_file_id',
            'other_file_id'])
        cut_df = cut_df.astype('object')

    # delete any cut_times previously there
    cut_time_cols = [col for col in cut_df.columns if col.startswith('cut_time')]
    cut_df = cut_df.drop(columns=cut_time_cols, errors='ignore')

    # Add cut_time columns to df
    for i in range(n_cuts):
        cut_df[f'cut_time{i+1}'] = None
        cut_df[f'cut{i+1}_dbo_id'] = 'pending'
        cut_df[f'cut{i+1}_other_id'] = 'pending'

    return cut_df


# Stub function to sync files with Box
# Integrate with Dominic with BoxAPI
def sync_with_box():
    logging.info("Syncing cut_list.csv and backup directory with Box.")

    # The location of cut_list.csv
    directory = "."

    # Upload local version of cut_list.csv to box. This will overwrite it
    logging.info(f"Uploading {cut_list_path} to box.")
    cut_list_path_id = box_functions.upload_to_box(directory, cut_list_path, song_list_root, client)
    logging.info(f"{cut_list_path} is now synced with box. Box File ID: {cut_list_path_id}")

    # Syncronize the backup directory "./bak_dir" with box
    logging.info(f"Uploading {bak_dir} to box.")
    for file_name in os.listdir(bak_dir):
        print(f"Syncronizing {file_name} with box ...")
        logging.info(f"Syncronizing {file_name} with box ...")
        # Call the upload_to_box function for each file
        bak_path_id = box_functions.upload_to_box(bak_dir, file_name, bak_folder_id, client)
        logging.info(f"{file_name} is now synced with box. Box File ID: {bak_path_id}")
    logging.info(f"{bak_dir} is now synced with box. Box Folder ID: {bak_folder_id}")

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
def update_cut_file_ids(file_name, dbo_id, drums_id, vocals_id, bass_id, other_id):
    if not os.path.exists(cut_list_path):
        logging.error("cut_list.csv not found. Cannot update file IDs.")
        return
    global df
    df = pd.read_csv(cut_list_path)
    
    # Update the file IDs for the specified entry
    df.loc[df['file_name'] == file_name, ['dbo_file_id', 'drums_file_id', 'vocals_file_id', 'bass_file_id', 'other_file_id']] = \
        [dbo_id, drums_id, vocals_id, bass_id, other_id]

    save_lookup_table(df, cut_list_path)
    logging.info(f"Updated file IDs for {file_name} in cut_list.csv.")

def gen_cuts(start_offset, cut_length, n_cuts):


    # Pull the song_list.csv from Box
    song_df = get_song_list()
    # print(f"song_df size: {song_df.shape}")

    # Prepare the cut_list DataFrame or load existing with specified dtypes
    # save globally in case of sigterm
    global df
    df = get_cut_df(n_cuts)
    # print(f"cut_df size: {cut_df.shape}")


    # Process each song in song_list.csv
    for _, row in song_df.iterrows():
        file_name = row['filename']
        file_id = row['box_file_id']
        song_length = row['songLength']
        # print(f"checking file_name: {file_name}")
        
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
            'dbo_file_id': 'pending',
            'drums_file_id': 'pending',
            'vocals_file_id': 'pending',
            'bass_file_id': 'pending',
            'other_file_id': 'pending'
        }
        for i, cut_time in enumerate(cut_times):
            cut_entry[f'cut_time{i+1}'] = str(cut_time)

        # Check if the entry already exists in cut_df
        if file_name in df['file_name'].values:
            # Get the index of the existing row
            row_index = df.index[df['file_name'] == file_name].tolist()[0]

            # Update each field in the existing row
            for key, value in cut_entry.items():
                # print(f"Updating '{key}' for '{file_name}': {value}")
                df.at[row_index, key] = value  # Directly set the cell value
            logging.info(f"Updated entry for {file_name} in cut_list.csv.")
        else:
            # Add as a new row if it does not exist
            df = pd.concat([df, pd.DataFrame([cut_entry])], ignore_index=True)
            logging.info(f"Added new entry for {file_name} in cut_list.csv.")

    # Save the updated cut_list.csv
    save_lookup_table(df, cut_list_path)


def save_lookup_table(data_frame, path):
    data_frame.to_csv(path, index=False)
    logging.info(f"Saved changes to {path}.")
    sync_with_box()


def remove_local_files(files_to_delete):
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Deleted {file_path}.")
        else:
            logging.info(f"Couldn't find {file_path}.")


def create_data_set():

    training_dir = "training"
    dbo_dir = os.path.join(training_dir, "dbo")
    other_dir = os.path.join(training_dir, "other")


    # make directories for training data
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    if not os.path.exists(dbo_dir):
        os.makedirs(dbo_dir)
    if not os.path.exists(other_dir):
        os.makedirs(other_dir)

    # assign to global in case of signal interrupt so we can save it
    global df
    df = get_cut_df(None)

    if df.empty:
        logging.error(f"cut_list.csv empty")
        sys.exit(1)
 

    generated_files = []
    batch_size = 100
    batch = 0
    for idx, row in df.iterrows():
        file_name = row['file_name']
        dbo_id = row['dbo_file_id']
        dbo_name = file_name.replace(".m4a", "_dbo.wav")
        other_id = row['other_file_id']
        other_name = file_name.replace(".m4a", "_other.wav")
        n_cuts = int(row['n_cuts'])

        if row['dbo_file_id'] == 'pending' or row['other_file_id'] == 'pending':
            logging.info(f"File {file_name} never preprocessed... skipping")
            continue


        if row['process_time'] != 'pending':
            logging.info(f"File {file_name} already preprocessed... skipping")
            continue

        # Step 1: retrieve dbo and other files from box

        num_attempts = 0
        successful = False
        while(not successful and num_attempts < 5):
            try:
                dbo_path = box_functions.download_from_box(".", dbo_name, dbo_id, client)
                other_path = box_functions.download_from_box(".", other_name, other_id, client)
                successful = True
            except Exception as e:
                num_attempts += 1
                logging.error(f"Could not download {dbo_name} or {other_name} attempt {num_attempts}: {e}")
        if not successful:
            logging.error(f"Could not download {dbo_name} or {other_name} possible connection error... saving and quitting")
            save_lookup_table(df, cut_list_path)
            sys.exit(1)
        # check for valid file paths 
        if (dbo_path == None) or (not os.path.exists(dbo_path)) or \
            (other_path == None) or (not os.path.exists(other_path)):
            logging.info(f"Couldn't download file {dbo_name} or {other_name} ... skipping")
            continue

        generated_files.append(dbo_path)
        generated_files.append(other_path)

        # iterate over n cuts 
        cut_length = row[f"cut_length"]
        for i in range(1, n_cuts + 1):
            cut_time = row[f"cut_time{i}"]
            
            # Create cut segment files for dbo and other files
            dbo_cut_name = dbo_name.replace(".wav", f"_cut{i}.wav")
            dbo_cut_file = os.path.join(dbo_dir, dbo_cut_name)
            
            other_cut_name = other_name.replace(".wav", f"_cut{i}.wav")
            other_cut_file = os.path.join(other_dir, other_cut_name)
            
            # FFMPEG to make cuts
            try:
                # subprocess.run(["ffmpeg", "-y", "-ss", str(cut_time), "-t", str(cut_length), "-i", dbo_path, dbo_cut_file], check=True)

                subprocess.run([
                                    "ffmpeg", 
                                    "-y", 
                                    "-ss", str(cut_time), 
                                    "-t", str(cut_length), 
                                    "-i", dbo_path,  # Input file
                                    "-ac", "1",  # Set audio to 1 channel (mono)
                                    "-ar", "24000",  # Set sampling rate to 24000 Hz
                                    dbo_cut_file  # Output file
                                ], check=True)

                # subprocess.run(["ffmpeg", "-y", "-ss", str(cut_time), "-t", str(cut_length), "-i", other_path, other_cut_file], check=True)

                subprocess.run([
                                    "ffmpeg", 
                                    "-y", 
                                    "-ss", str(cut_time), 
                                    "-t", str(cut_length), 
                                    "-i", other_path,  # Input file
                                    "-ac", "1",  # Set audio to 1 channel (mono)
                                    "-ar", "24000",  # Set sampling rate to 24000 Hz
                                    other_cut_file  # Output file
                                ], check=True)



            except subprocess.CalledProcessError as e:
                logging.error(f"Error during ffmpeg processing: {e}")
                continue  # Skip this cut
            except Exception as e:
                logging.error(f"Some other error {e} exiting...")
                save_lookup_table(df, cut_list_path)
                sys.exit(1)


            # Upload cut segment files to Box and store file IDs
            try:
                dbo_cut_id = box_functions.upload_to_box(".", dbo_cut_file, training_cuts_dbo_id, client)
                other_cut_id = box_functions.upload_to_box(".", other_cut_file, training_cuts_other_id, client)
            except Exception as e:
                logging.error(f"Error uploading cut files to Box: {e}")
                save_lookup_table(df, cut_list_path)
                continue  # Skip to next cut


            # Update df with cut segment file IDs for dbo and other files
            df.loc[idx, f'cut{i}_dbo_id'] = dbo_cut_id
            df.loc[idx, f'cut{i}_other_id'] = other_cut_id

        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.loc[idx, "process_time"] = str(upload_time)

        if (generated_files):
            remove_local_files(generated_files)
            generated_files.clear()

        logging.info(f"Finished Preprocessing {file_name}")
        batch += 1

        if (batch >= batch_size):
            logging.info("cut_list.csv saving current batch.")
            save_lookup_table(df, cut_list_path)
            logging.info("cut_list.csv finished saving current batch.")
            batch = 0


    # Save the updated cut_list.csv and sync it with Box
    save_lookup_table(df, cut_list_path)
    logging.info("cut_list.csv has been updated with new cut segment times and file IDs.")


def get_processed_ids():
    global df
    df = get_cut_df(None)

    if df.empty:
        logging.error(f"cut_list.csv empty")
        return

    # dbo_file_id,drums_file_id,vocals_file_id,bass_file_id,other_file_id

    # bass_folder_id = '292547961183'

    folders = {"dbo": dbo_folder_id, "drums": drums_folder_id, "vocals": vocals_folder_id, "bass": bass_folder_id, "other": other_folder_id}
    
    # get all uploaded files in all of the above folders
    all_files = {}
    for folder, id in folders.items():
        folder_id = id
        items = box_functions.get_all_items(client, folder_id)
        # remove any duplicates which may happen when requesting files from box
        unique_items = list({item.id: item for item in items}.values())
        logging.info(f"Got {len(items)} items from Box...")
        logging.info(f"There's {len(unique_items)} unique items")
        for file in unique_items:
            all_files[file.name] = file.id
            

    # iterate through the dataframe and update the ids of the ids
    updates_made = False
    for idx, row in df.iterrows():
        filename = row['file_name']
        #dbo_file_id,drums_file_id,vocals_file_id,bass_file_id,other_file_id
        dbo_name = filename.replace(".m4a", "_dbo.wav")
        drums_name = filename.replace(".m4a", "_drums.wav")
        vocals_name = filename.replace(".m4a", "_vocals.wav")
        bass_name = filename.replace(".m4a", "_bass.wav")
        other_name = filename.replace(".m4a", "_other.wav")

        updates = {}

        if row['dbo_file_id'] == 'pending' and dbo_name in all_files:
            updates['dbo_file_id'] = all_files[dbo_name]
        if row['drums_file_id'] == 'pending' and drums_name in all_files:
            updates['drums_file_id'] = all_files[drums_name]
        if row['vocals_file_id'] == 'pending' and vocals_name in all_files:
            updates['vocals_file_id'] = all_files[vocals_name]
        if row['bass_file_id'] == 'pending' and bass_name in all_files:
            updates['bass_file_id'] = all_files[bass_name]
        if row['other_file_id'] == 'pending' and other_name in all_files:
            updates['other_file_id'] = all_files[other_name]
        
        if len(updates) > 0: #updates made
            logging.info(f"Uploading {len(updates)} for {filename}")
            for key, value in updates.items():
                df.loc[idx, key] = value
            updates_made = True
            
    if updates_made:
        logging.info("Updated cut_list.csv with missing file IDs. Uploading to Box")
        save_lookup_table(df, cut_list_path)
        logging.info("cut_list.csv has been updated with new cut segment times and file IDs.")
    else:
        logging.info("No updates were made")

def remix_files():
    global df
    df = get_cut_df(None)

    if df.empty:
        logging.error(f"cut_list.csv empty")
        return

    training_dir = "training"
    folders = {"dbo": training_cuts_dbo_id, "other": training_cuts_other_id}
    
    # reset process_time to 'pending' for all files
    df['process_time'] = 'pending'

    # get all uploaded files in all of the above folders
    all_files = {}
    batch = 0
    for folder, id in folders.items():
        folder_id = id
        file_dir = os.path.join(training_dir, folder)   # e.g. training/dbo or training/other
        items = box_functions.get_all_items(client, folder_id)
        # remove any duplicates which may happen when requesting files from box
        unique_items = list({item.id: item for item in items}.values())
        logging.info(f"Got {len(items)} items from Box...")
        logging.info(f"There's {len(unique_items)} unique items")
        
        for file in unique_items:
            file_path = os.path.join(file_dir, file.name)

            if not os.path.exists(file_path):
                logging.info(f"couldn't find file {file_path}")
                continue

            try:
                subprocess.run([
                                    "ffmpeg", 
                                    "-y", 
                                    "-i", file_path,  # Input file
                                    "-ac", "1",       # Mix to mono
                                    "-ar", "24000",   # Resample to 24000 Hz
                                    "temp.wav"        # Temporary output file
                                ], check=True)

                                # Replace the original file with the temporary file
                shutil.move("temp.wav", file_path)
            except Exception as e:
                print(f"Error processing file with FFmpeg: {e}")
            finally:
                if os.path.exists("temp.wav"):
                    os.remove("temp.wav")
            
            # reupload
            successful = False
            num_attempts = 0
            while (not successful and num_attempts < 5):
                try:
                    logging.info(f"Updating {file.name} on box ...")
                    uploaded_file = client.file(file.id).update_contents(file_path)
                    print(f"{file.name} Updated")
                    successful = True
                except Exception as e:
                    num_attempts += 1
                    logging.error(f"Could not upload {file.name} attempt {num_attempts}: {e}")
                    time.sleep(2)
            if not successful:
                logging.info(f"Could not upload {file.name} possible connection error... saving and quitting")
                sys.exit(1)

            chop_length = (len(folder) + 10) * -1
            lookup_id = file.name[:chop_length]+".m4a" # remove _dbo_cut1.wav or _other_cut1.wav, add m4a
            upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df.loc[df['file_name'] == lookup_id, ['process_time']] = upload_time
            
            logging.info(f"Finished Remixing {file.name}")
            
            batch += 1

            if (batch >= 100):
                logging.info("cut_list.csv saving current batch.")
                save_lookup_table(df, cut_list_path)
                logging.info("cut_list.csv finished saving current batch.")
                batch = 0

        logging.info(f"finished folder {folder}")
        save_lookup_table(df, cut_list_path)
   


def preprocess_audio(n_cuts):

    generated_files = []
    directory = "serv_delete"

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Retrieve the latest cut_list.csv and save globally for term_signal
    global df
    df = get_cut_df(n_cuts)


    # Process each file in cut_df
    batch_size = 100
    batch = 0   # upload after batch of 100
    for _, row in df.iterrows():
        file_name = row['file_name']
        file_id = row['file_id']
        
        if row['dbo_file_id'] != 'pending':
            logging.info(f"File {file_name} already preprocessed... skipping")
            continue

        if len(generated_files) != 0:
            remove_local_files(generated_files)
            generated_files.clear()

        # song_length = float(row['song_length'])
        # start_offset = int(row['start_offset'])
        # cut_length = int(row['cut_length'])
        # n_cuts = int(row['n_cuts'])
        print(f"working on file {file_name}")
        # logging(f"working on file {str(file_name)}")

        # Step 1: retrieve base file from box
        local_path = box_functions.download_from_box(directory, file_name, file_id, client)
        if (local_path == None) or (not os.path.exists(local_path)):

            # search by name
            existing_file = box_functions.check_existing(file_name, client, orig_folder_id)
            if not existing_file:
                # still didn't find
                logging.info(f"Error getting file {file_name}... skipping")
                continue

            # update file_id in song_list and cut_list:

            local_path = box_functions.download_from_box(directory, file_name, existing_file.id, client)

            if (local_path == None) or (not os.path.exists(local_path)):
                logging.info(f"Still couldn't download file {file_name} with old id {file_id} or {existing_file.id}... skipping")
                continue

            logging.info(f"Found existing file with diff id {file_name}, song_list.csv out of date")
            logging.info(f"Old Info: filename: {file_name}, id: {file_id}")
            logging.info(f"New Info: filename: {existing_file.name}, {existing_file.id}")
            
            file_id = existing_file.id

        print(f"downloaded file to {local_path}")
        # logging(f"downloaded file to {local_path}")
        generated_files.append(local_path)

        # Step 2: Convert .m4a to .wav
        wav_file = local_path.replace(".m4a", ".wav")

        # first make wav with 44.1khz sr and s16p bit depth
        subprocess.run(["ffmpeg", "-i", local_path, "-ar", "44100", "-sample_fmt", "s16", wav_file])
        if not os.path.exists(wav_file):
            logging.info(f"Error preprocessing {file_name} with FFMPEG... skipping")
            continue
        generated_files.append(wav_file)

        # Step 3: Run demucs.py
        demucs_output = [f"{wav_file.replace('.wav', suffix)}.wav" for suffix in ["_dbo", "_drums", "_vocals", "_bass", "_other"]]
        subprocess.run(["python3", "pp_demucs.py", wav_file, "--device", "cuda"])
        
        isDemucsSuccessful = True
        for file in demucs_output:
            if not os.path.exists(file):
                logging.info(f"{file} from demucs doesn't exist... skipping")
                isDemucsSuccessful = False
                break
            else:
                generated_files.append(file)
        
        if not isDemucsSuccessful:
            continue

        # Step 4: Upload demucs output to Box and update cut_df
        file_ids = {}
        folder_map = {
            '_dbo': dbo_folder_id,
            '_drums': drums_folder_id,
            '_vocals': vocals_folder_id,
            '_bass': bass_folder_id,
            '_other': other_folder_id
        }
        
        for demucs_file in demucs_output:

            # mix again down to 

            file_type = demucs_file.split('_')[-1].replace(".wav", "")
            folder_id = folder_map.get(f"_{file_type}")
        
            num_attempts = 0
            successful = False
            while(not successful and num_attempts < 5):
                try:
                    file_id = box_functions.upload_to_box(directory, os.path.basename(demucs_file), folder_id, client)
                    successful = True
                except Exception as e:
                    num_attempts += 1
                    logging.error(f"Could not upload {demucs_file} attempt {num_attempts}: {e}")
            if not successful:
                logging.info(f"Could not upload {demucs_file} possible connection error... saving and quitting")
                save_lookup_table(df, cut_list_path)
                logging.info("cut_list.csv has been updated but preprocessing is incomplete.")
                sys.exit(1)



            file_ids[f"{file_type}_file_id"] = file_id  # Store the file_id for updating cut_df

        # Update cut_df with demucs output file IDs
        df.loc[df['file_name'] == file_name, ['dbo_file_id', 'drums_file_id', 'vocals_file_id', 'bass_file_id', 'other_file_id']] = \
            [file_ids['dbo_file_id'], file_ids['drums_file_id'], file_ids['vocals_file_id'], file_ids['bass_file_id'], file_ids['other_file_id']]

        # Step 5: Generate cuts for each specified cut time

        # for i in range(1, n_cuts + 1):
        #     cut_time = int(row[f'cut_time{i}'])
            
        #     # Create cut segment files for dbo and other files
        #     dbo_cut_file = f"{file_name.replace('.m4a', '')}_dbo_cut{i}.wav"
        #     other_cut_file = f"{file_name.replace('.m4a', '')}_other_cut{i}.wav"
            
        #     # ffmpeg command to generate cuts
        #     subprocess.run(["ffmpeg", "-ss", str(cut_time), "-t", str(cut_length), "-i", demucs_output[0], dbo_cut_file])
        #     subprocess.run(["ffmpeg", "-ss", str(cut_time), "-t", str(cut_length), "-i", demucs_output[4], other_cut_file])

        #     # Upload cut segment files to Box and store file IDs
        #     dbo_cut_id = box_functions.upload_to_box('.', dbo_cut_file, training_data_folder_id, client)
        #     other_cut_id = box_functions.upload_to_box('.', other_cut_file, training_data_folder_id, client)

        #     # Update df with cut segment file IDs for dbo and other files
        #     df.at[df['file_name'] == file_name, f'cut{i}_dbo_id'] = dbo_cut_id
        #     df.at[df['file_name'] == file_name, f'cut{i}_other_id'] = other_cut_id

        logging.info(f"Finished Preprocessing {file_name}")
        batch += 1

        if (batch >= batch_size):
            logging.info("cut_list.csv saving current batch.")
            save_lookup_table(df, cut_list_path)
            logging.info("cut_list.csv finished saving current batch.")
            batch = 0

    if len(generated_files) != 0:
        remove_local_files(generated_files)
        generated_files.clear()

    # Step 5: Save the updated cut_list.csv and sync it with Box
    save_lookup_table(df, cut_list_path)
    logging.info("cut_list.csv has been updated with new cut segment times and file IDs.")

# Main function with argument parsing
def main():
    parser = argparse.ArgumentParser(description="Manage and preprocess audio track cuts using cut_list.csv.")
    parser.add_argument(
        '--mode', choices=['gencuts', 'preprocess', 'create_data'], required=True,
        help="Operation mode: 'gencuts' to create/update cut_list, 'preprocess' to process audio based on cut_list, create_data to download/create files from cut times"
    )
    parser.add_argument('--start_offset', type=int, default=None, help="Start offset for random cuts.")
    parser.add_argument('--length', type=int, default=None, help="Length of each cut.")
    parser.add_argument('--n_cuts', type=int, default=None, help="Number of cuts per audio file.")
    args = parser.parse_args()

    # Execute based on mode
    cut_config = read_cut_config()
    if args.mode == 'gencuts':

        start_offset = args.start_offset if args.start_offset is not None else cut_config['Start_offset']
        cut_length = args.length if args.length is not None else cut_config['length']
        n_cuts = args.n_cuts if args.n_cuts is not None else cut_config['n_cuts']

        gen_cuts(start_offset, cut_length, n_cuts)
    elif args.mode == 'preprocess':
        preprocess_audio(cut_config['n_cuts'])

    elif args.mode == 'create_data':
        create_data_set()
if __name__ == "__main__":
    main()
