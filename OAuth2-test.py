import os
import json
import configparser
from boxsdk import OAuth2, Client
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

# Load Box credentials from config.ini
def load_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config['box']['client_id'], config['box']['client_secret'], config['box']['redirect_uri']

# Load tokens from a file (if they exist)
def load_tokens(token_file='tokens.json'):
    if os.path.exists(token_file):
        with open(token_file, 'r') as file:
            tokens = json.load(file)
            return tokens.get('access_token'), tokens.get('refresh_token')
    return None, None

# Save tokens to a file
def save_tokens(access_token, refresh_token, token_file='tokens.json'):
    with open(token_file, 'w') as file:
        json.dump({'access_token': access_token, 'refresh_token': refresh_token}, file)

# Temporary server to handle OAuth 2.0 authorization
class OAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.server.auth_code = self.path.split('code=')[-1]
        self.wfile.write(b'OAuth 2.0 Authorization Completed! You may close this window.')

# Step 1: Authenticate the user with OAuth 2.0, using refresh token if available
def authenticate_with_oauth(client_id, client_secret, redirect_uri):
    # Check if we already have stored tokens (access token and refresh token)
    access_token, refresh_token = load_tokens()

    # If tokens exist, use them
    oauth2 = OAuth2(
        client_id=client_id,
        client_secret=client_secret,
        access_token=access_token,
        refresh_token=refresh_token,
    )

    if access_token is None:
        # No stored access token, so start the OAuth flow
        print("No access token found. Starting OAuth flow.")
        auth_url, csrf_token = oauth2.get_authorization_url(redirect_uri)
        webbrowser.open(auth_url)

        # Start a simple HTTP server to listen for the OAuth callback
        httpd = HTTPServer(('localhost', 8080), OAuthHandler)
        httpd.handle_request()

        # Once authorized, get the authorization code
        auth_code = httpd.auth_code

        # Exchange the authorization code for access/refresh tokens
        access_token, refresh_token = oauth2.authenticate(auth_code)

        # Save the tokens to a file
        save_tokens(access_token, refresh_token)
    else:
        print("Using stored tokens.")
        # Refresh the access token if it's expired
        oauth2.refresh(access_token)

    return Client(oauth2)

# Step 2: Upload or update files in the Box folder
def upload_files_to_box(client, folder_id, local_folder_path):
    # Get the folder object
    folder = client.folder(folder_id)
    
    # List local files and upload them
    for file_name in os.listdir(local_folder_path):
        local_file_path = os.path.join(local_folder_path, file_name)
        
        # Upload the file
        with open(local_file_path, 'rb') as file_stream:
            folder.upload_stream(file_stream, file_name)
        print(f"Uploaded {file_name} to Box.")


if __name__ == "__main__":
    # Load credentials from config file
    client_id, client_secret, redirect_uri = load_config()

    # Step 1: Authenticate and create Box client (using refresh token if available)
    client = authenticate_with_oauth(client_id, client_secret, redirect_uri)

    # Box folder ID and local folder path
    BOX_FOLDER_ID = '287513576583'
    LOCAL_FOLDER_PATH = '../output'

    # Step 2: Upload files to Box folder
    upload_files_to_box(client, BOX_FOLDER_ID, LOCAL_FOLDER_PATH)
