import os
from boxsdk.exception import BoxAPIException

# Get ALL files from a box folder
# Allows for more than 100 files to be searched
def get_all_items(client, folder_id):
    items = []
    offset = 0
    limit = 1000  # Maximum allowed by Box API

    while True:
        # Retrieve a batch of items
        batch = list(client.folder(folder_id).get_items(limit=limit, offset=offset, fields=['type', 'id', 'name', 'created_at']))
        
        # Add the batch to the items list
        items.extend(batch)
        
        # Break the loop if fewer items were returned than the limit, indicating the end
        if len(batch) < limit:
            break
        
        # Increase offset to get the next batch
        offset += limit

    return items

def check_existing(directory, file_name, client, folder_id):
    items = get_all_items(client, folder_id)
    existing_file = None

    for item in items:
        if item.name == file_name:
            print(f"File '{file_name}' exists in the folder.")
            existing_file = item
            break
    return existing_file

# Uploads a file to the box cloud storage
# Requires path, file, parent folder id, and client object
def upload_to_box(directory, file_name, folder_id, client):
    # Get local path to file
    file_path = os.path.join(directory, file_name)

    # Determine if the file already exists on box
    existing_file = get_existing(directory, file_name, client, folder_id)
        
    if existing_file:
        # Update the file if it exists on box
        print(f"Updating {file_name} on box ...")
        uploaded_file = existing_file.update_contents(file_path)
        print(f"{file_name} updated.")
    else:
        # Upload the file to box if it does not exists on box
        print(f"File '{file_name}' does not exist in the folder. Creating file ...")
        uploaded_file = client.folder(folder_id).upload(file_path)
        print(f"{file_name} uploaded.")

    # return the created Folder ID
    return uploaded_file.id   

# Downloads a file from the box cloud storage to a local directory
# Requires path, name of the file, box file id, and client object
def download_from_box(directory, file_name, file_id, client):
    # Create the path for the download
    file_path = os.path.join(directory, file_name)

    # Determine if the file exists on box before proceeding
    try:
        # Attempt to retrieve the file metadata to check if it exists
        file_info = client.file(file_id).get()
    except BoxAPIException as e:
        if e.status == 404:
            print(f"File with ID {file_id} not found on Box.")

            return None  # File doesn't exist on Box
        else:
            raise  # Re-raise other exceptions

    # Check if the file already exists locally
    if os.path.exists(file_path):
        print(f"File {file_name} exists locally. It will be overwritten.")

    # Download/Overwrite the file
    with open(file_path, 'wb') as file:
        client.file(file_id).download_to(file)

    print(f"Downloaded {file_name} to {file_path}")
    return file_path


# Deletes a file from box using the file id
def delete_from_box(file_id, client):
    try:
        # Try to get the file metadata to check if it exists
        file_info = client.file(file_id).get()
        print(f"File '{file_info.name}' found in Box.")

        # If the file exists, delete it
        client.file(file_id).delete()
        print(f"File '{file_info.name}' has been deleted from Box.")

    except Exception as e:
        # Output the error if the file does not exist or another error occurs
        print(f"Error deleting file with ID '{file_id}': {e}")
        return False  # Return False if there was an error

    return True  # Return True if the file was successfully deleted
