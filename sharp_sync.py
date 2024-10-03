import os
from boxsdk import Client, JWTAuth
from boxsdk.exception import BoxAPIException

# Path to the Box JWT config file
JWT_CONFIG_PATH = '../30668568_pykuu5d0_config.json'

# Box folder ID where you want to upload files (you can find this in the URL of the folder)
BOX_FOLDER_ID = '287513576583'

LOCAL_FOLDER_PATH = '../output'

# Authenticate with Box using JWT
def authenticate():
    auth = JWTAuth.from_settings_file(JWT_CONFIG_PATH)
    client = Client(auth)
    return client

# Get list of files in a Box folder
def get_box_files(client, folder_id):
    folder = client.folder(folder_id).get()
    items = folder.get_items()
    box_files = {item.name: item.id for item in items}
    return box_files

# Upload or update a file in Box
def upload_or_update_file(client, folder_id, local_file_path, box_file_id=None):
    folder = client.folder(folder_id)
    file_name = os.path.basename(local_file_path)
    
    with open(local_file_path, 'rb') as file_stream:
        if box_file_id:
            # File exists in Box, update it
            try:
                print(f"Updating file {file_name} in Box.")
                file = client.file(box_file_id).update_contents(file_stream)
            except BoxAPIException as e:
                print(f"Failed to update {file_name}: {e}")
        else:
            # File does not exist, upload new
            try:
                print(f"Uploading new file {file_name} to Box.")
                folder.upload_stream(file_stream, file_name)
            except BoxAPIException as e:
                print(f"Failed to upload {file_name}: {e}")

# Sync local folder to Box folder
def sync_folder_to_box(client, local_folder_path, box_folder_id):
    # Get list of existing files in Box folder
    box_files = get_box_files(client, box_folder_id)

    # Iterate through local folder and upload/update files
    for root, _, files in os.walk(local_folder_path):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            box_file_id = box_files.get(file_name)  # Check if file exists in Box
            upload_or_update_file(client, box_folder_id, local_file_path, box_file_id)

if __name__ == "__main__":
    # Authenticate and create Box client
    client = authenticate()

    # Sync the local folder with the Box folder
    sync_folder_to_box(client, LOCAL_FOLDER_PATH, BOX_FOLDER_ID)
