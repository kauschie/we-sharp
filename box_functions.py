import os
from boxsdk import Client, JWTAuth
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Set up JWT authentication
auth = JWTAuth.from_settings_file(os.getenv("KEYPAIR_JSON_PATH"))
client = Client(auth)

# Function to upload files to Box folder
def upload_files_to_box(folder_id, local_folder_path):

    # Check if the provided path is valid
    if not os.path.exists(local_folder_path) or not os.path.isdir(local_folder_path):
        # Exit the function if the path is not valid
        print(f'Error: "{local_folder_path}" is not a valid directory path.')
        return


    # Uploads all folders and files from the local directory to the specified Box folder.
    for folder_name in os.listdir(local_folder_path):
        folder_path = os.path.join(local_folder_path, folder_name)

        if os.path.isdir(folder_path):
            print(f'Processing folder: {folder_name}')

            # Check if folder already exists in Box
            items = client.folder(folder_id).get_items()
            existing_folder = next((item for item in items if item.name == folder_name), None)

            if existing_folder:
                print(f'Folder "{folder_name}" already exists on Box with ID: {existing_folder.id}')
                continue
            else:
                # Create a new folder on Box
                box_folder = client.folder(folder_id).create_subfolder(folder_name)
                print(f'Created folder on Box: {box_folder.name} with ID: {box_folder.id}')

            # Upload files inside the local folder
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                try:
                    # Upload the file to the newly created Box folder
                    uploaded_file = client.folder(box_folder.id).upload(file_path)
                    print(f'Successfully uploaded "{uploaded_file.name}" with file ID: {uploaded_file.id}')
                except Exception as e:
                    print(f'Error uploading "{file_name}": {e}')

    print(f'\nFinished uploading from {local_folder_path}.\n')

# Function to download files from Box folder
def download_files_from_box(folder_id, local_folder_path):

    # Ensure the local directory exists
    if not os.path.exists(local_folder_path):
        os.makedirs(local_folder_path)
        print(f'Created local folder: {local_folder_path}')

    # Get items in the Box folder
    items = client.folder(folder_id).get_items()

    for item in items:
        if item.type == 'folder':
            # Create a corresponding folder in the local directory
            local_subfolder_path = os.path.join(local_folder_path, item.name)
            if not os.path.exists(local_subfolder_path):
                os.makedirs(local_subfolder_path)
                print(f'Created local folder: {local_subfolder_path}')
            else:
                print(f'Local folder {local_subfolder_path} already exists')
            # Recursively download files from the subfolder
            download_files_from_box(item.id, local_subfolder_path)

        elif item.type == 'file':
            # Download the file
            file_path = os.path.join(local_folder_path, item.name)
            if not os.path.exists(file_path):
                with open(file_path, 'wb') as f:
                    print(f'Downloading file: {item.name}')
                    file_content = item.download_to(f)
                print(f'Successfully downloaded: {item.name}')
            else:
                print(f'Local file {item.name} already exists')
    print(f'\nFinished downloading to {local_folder_path}.\n')

# Function to delete folders from Box
def DeleteFromBox(folder_id):

    # Deletes a folder from Box based on the folder ID.
    try:
        client.folder(folder_id).delete()
        print(f'Folder {folder_id} deleted successfully.\n')
    except Exception as e:
        print(f'Error deleting folder {folder_id}: {e}')

# Main function
def main():

    # Folder ID from which files will be uploaded/downloaded
    box_folder_id = os.getenv("FOLDER_ID")

    user_input = 0

    # Repeat program until user exits the program
    while user_input != '4':

        # User Prompt
        print("\nWhat operation would you like to perform?")
        print("=========================================\n")

        # Menu Options
        print("1 - Upload directory/folder to Box\n")
        print("2 - Download directory/folder from Box\n")
        print("3 - Delete directory/folder from Box\n")
        print("4 - Ends the program\n")

        # Get user input
        user_input = input("Waiting for user input: ")
        print("\n")

        if user_input == '1':
            # Prompt user for the local directory to upload files from
            local_dir = input("Please enter the local directory that you would like to upload to Box: ")
            # Begin upload function
            upload_files_to_box(box_folder_id, local_dir)

        elif user_input == '2':
            # Prompt user for the local directory to download files to
            local_dir = input("Please enter the directory that you would like to download to: ")
            # Begin download function
            download_files_from_box(box_folder_id, local_dir)

        elif user_input == '3':
            # Prompt user for the Box folder ID to delete
            folder_id = input("Please enter the Box folder ID you would like to delete: ")
            # Begin delete function
            DeleteFromBox(folder_id)

        elif user_input == '4':
            print("Goodbye!")

        else:
            print("Invalid selection!\n\n")

# Call the main function
if __name__ == "__main__":
    main()
