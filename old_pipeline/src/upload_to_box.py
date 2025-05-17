import os
import argparse
from boxsdk import Client, JWTAuth
import box_functions

# Constants
BOX_FOLDER_ID = '313442530842'  # Replace if needed (default: music folder from server.py)

def main():
    parser = argparse.ArgumentParser(description="Upload a file to Box using a given file path.")
    parser.add_argument("file", type=str, help="Path to the local file to upload.")
    parser.add_argument("--folder_id", type=str, default=BOX_FOLDER_ID,
                        help="Box folder ID to upload the file into. Defaults to music folder.")

    args = parser.parse_args()
    file_path = args.file
    folder_id = args.folder_id

    # Validate file path
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    # Extract directory and filename
    directory, file_name = os.path.split(file_path)

    # Authenticate with Box using JWT
    auth = JWTAuth.from_settings_file('./keypair.json')
    client = Client(auth)

    # Upload the file
    try:
        uploaded_file_id = box_functions.upload_to_box(directory, file_name, folder_id, client)
        print(f"Upload successful. Box File ID: {uploaded_file_id}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
