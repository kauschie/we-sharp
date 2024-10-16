# Uploads local wav files to we-sharp/music (Box directory)

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

# Get the list of files in the Box folder
items = client.folder(folder_id).get_items()

# Create a set of filenames already present in the Box folder
box_file_names = {item.name for item in items}

# Directory containing WAV files
music_dir = '../music/wav'

# Loop through all the files in the directory and filter for WAV files
for file_name in os.listdir(music_dir):
    
    if file_name.endswith('.wav'):  # Check if the file is a WAV file

        # Skip if the file already exists in Box
        if file_name in box_file_names:
            print(f'Skipping "{file_name}" as it already exists in Box')
            continue

        # Get the full path of the file
        file_path = os.path.join(music_dir, file_name)

        try:
            # Upload the WAV file to the Box folder
            print(f'Uploading "{file_name}"...')
            uploaded_file = client.folder(folder_id).upload(file_path)
            
            # Confirm successful upload
            print(f'Successfully uploaded "{uploaded_file.name}" with file ID {uploaded_file.id}')

        except Exception as e:
            # Handle upload errors (e.g., network issues, API errors)
            print(f'Error uploading "{file_name}": {e}')