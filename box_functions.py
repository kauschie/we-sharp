import os

# Get ALL files from a box folder
# Allows for more than 100 files to be searched
def get_all_items(client, folder_id):
    items = []
    offset = 0
    limit = 100  # Maximum allowed by Box API

    # while True:
        # Retrieve a batch of items
        # batch = client.folder(folder_id).get_items(limit=limit, offset=offset)
    batch = client.folder(folder_id).get_items(limit=limit, offset=offset)
        
        # Add the batch to the items list
        # items.extend(batch)
        
        # Break the loop if fewer items were returned than the limit, indicating the end
        # if len(batch) < limit:
        #     break
        
        # Increase offset to get the next batch
        # offset += limit

    return items

# Uploads a file to the box cloud storage
# Requires path, file, parent folder id, and client object
def upload_to_box(directory, file_name, folder_id, client):
    # Get local path to file
    file_path = os.path.join(directory, file_name)

    # Upload the file to box
    uploaded_file = client.folder(folder_id).upload(file_path)

    # return the created Folder ID
    return uploaded_file.id   

# Downloads a file from the box cloud storage to a local directory
# Requires path, file, parent folder id, and client object
def download_from_box(directory, file_name, folder_id, client):

    # items = get_all_items(client, folder_id)
    items = client.folder(folder_id).get_items()

    for item in items:
        if item.name == file_name:
            # File found, proceed to download
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'wb') as file:
                client.file(item.id).download_to(file)
                # file.write(file_content)

            print(f"Downloaded {file_name} to {file_path}")
            return file_path

    print(f"File '{file_name}' not found in the specified Box folder.")

# Deletes a file from box using the name of the file and the parent folder id
def delete_from_box(file_name, folder_id, client):

    # Search for the file by name in the specified folder
    items = get_all_items(client, folder_id)
    
    for item in items:
        if item.name == file_name and item.type == 'file':
            # If the file is found, delete it
            item.delete()
            print(f"Deleted file: {file_name}")
            return True
    
    print(f"File '{file_name}' not found in folder {folder_id}.")
    return False
