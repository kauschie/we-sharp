# Exports music files to a Box Directory: "music"
# Uploads files and their respective directories

import os
from boxsdk import Client, JWTAuth
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Set up JWT authentication
auth = JWTAuth.from_settings_file(os.getenv("KEYPAIR_JSON_PATH"))

client = Client(auth)

# Folder ID where files will be uploaded
folder_id = os.getenv("FOLDER_ID")

# Path to the playlists directory
playlists_dir = '../music/playlists'

# Loop through each folder in the playlists directory
for folder_name in os.listdir(playlists_dir):

    folder_path = os.path.join(playlists_dir, folder_name)

    # Check if it's a directory
    if os.path.isdir(folder_path):

        print(f'Processing folder: {folder_name}')

        # Check if the folder already exists in "music" directory
        items = client.folder(folder_id).get_items()
        existing_folder = next((item for item in items if item.name == folder_name), None)

        if existing_folder:
            print(f'Folder "{folder_name}" already exists on Box with ID: {existing_folder.id}')
            # Skip to next folder
            continue
        else:
            # Create a corresponding folder on Box
            box_folder = client.folder(folder_id).create_subfolder(folder_name)
            print(f'Created folder on Box: {box_folder.name} with ID: {box_folder.id}')


        # Loop through each file in the local folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.mp3', '.m4a', '.wav')):  # Check for specific audio file types

                file_path = os.path.join(folder_path, file_name)  # Full path of the file
                try:
                    # Upload the file to the Box folder
                    uploaded_file = client.folder(box_folder.id).upload(file_path)
                    print(f'Successfully uploaded "{uploaded_file.name}" with file ID: {uploaded_file.id}')
                except Exception as e:
                    print(f'Error uploading "{file_name}": {e}')

        print(f'Finished processing folder: {folder_name}\n')

print(f"Program is finished. Check results here: https://csub.app.box.com/folder/{folder_id}")
