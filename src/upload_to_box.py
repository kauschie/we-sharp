import os
import argparse
from boxsdk import Client, JWTAuth
import box_functions

# Constants
DEFAULT_BOX_FOLDER_ID = '313442530842'  # Replace if needed (default: music folder from server.py)

def handle_path(path, folder_id, client):
    if os.path.isfile(path):
        directory, file_name = os.path.split(path)
        box_functions.upload_to_box(directory, file_name, folder_id, client)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for name in files:
                full_path = os.path.join(root, name)
                print(f"Uploading {full_path} ...")
                directory, file_name = os.path.split(full_path)
                try:
                    box_functions.upload_to_box(directory, file_name, folder_id, client)
                except Exception as e:
                    print(f"error uploading {filename}: {e}")
    else:
        print(f"Invalid path: {path}")



def main():
    parser = argparse.ArgumentParser(description="Upload a file or directory to Box.")
    parser.add_argument("path", type=str, help="Path to file or directory to upload.")
    parser.add_argument("--folder_id", type=str, default=DEFAULT_BOX_FOLDER_ID, help="Box folder ID to upload into (default is music folder).")


    args = parser.parse_args()
    
    auth = JWTAuth.from_settings_file('./keypair.json')
    client = Client(auth)

    handle_path(args.path, args.folder_id, client)

if __name__ == "__main__":
    main()
