import requests
import base64
import json
import pandas as pd
import logging
import time
from fuzzywuzzy import fuzz  # Import fuzzy matching library

# Set up logging to separate info and warnings/errors
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Handler for INFO (and above) messages (success logs)
info_handler = logging.FileHandler('info.log')
info_handler.setLevel(logging.INFO)
info_formatter = logging.Formatter('%(message)s')  # Simplified log output
info_handler.setFormatter(info_formatter)

# Handler for WARNING (and above) messages (warnings/errors logs)
warn_handler = logging.FileHandler('warnings.log')
warn_handler.setLevel(logging.WARNING)
warn_formatter = logging.Formatter('%(message)s')  # Simplified log output
warn_handler.setFormatter(warn_formatter)

# Add handlers to the logger
logger.addHandler(info_handler)
logger.addHandler(warn_handler)

# Function to split list into batches of n items
def batch_track_uris(track_uris, batch_size=100):
    for i in range(0, len(track_uris), batch_size):
        yield track_uris[i:i + batch_size]

# Function to add tracks to a playlist in batches of 100
def add_tracks_to_playlist(token, track_uris, playlist_id):
    add_tracks_url = f"{API_URL}/playlists/{playlist_id}/tracks"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # Split track URIs into batches of 100
    for batch in batch_track_uris(track_uris, 100):
        data = json.dumps({'uris': batch})
        add_response = requests.post(add_tracks_url, headers=headers, data=data)
        if add_response.status_code == 201:
            logger.info(f"Added {len(batch)} tracks to playlist ID {playlist_id}")
        else:
            logger.error(f"Error adding tracks: {add_response.status_code} - {add_response.json()}")

# Step 6: Search for tracks and add them to the playlist with artist and track name fuzzy fallback
def search_and_add_tracks_to_playlist(token, csv_file_path, playlist_id):
    df = pd.read_csv(csv_file_path, header=None)
    df.columns = ['Item', 'Track', 'Artist', 'Top10', 'Year']
    
    track_uris = []
    for index, row in df.iterrows():
        item_number = row['Item']
        track_title = row['Track']
        artist_name = row['Artist']
        track_year = row['Year']  # Year from the CSV

        # Step 1: Try to search by track and artist first for an exact match
        search_url = f"{API_URL}/search"
        headers = {
            'Authorization': f'Bearer {token}'
        }
        params = {
            'q': f"track:{track_title} artist:{artist_name}",  # Search by track title and artist
            'type': 'track',
            'limit': 1  # Limit to 1 for exact match attempts
        }

        response = requests.get(search_url, headers=headers, params=params)
        if response.status_code == 200:
            tracks = response.json()['tracks']['items']
            if tracks:
                # Exact match found, add it directly
                track_uris.append(tracks[0]['uri'])
                logger.info(f"Exact match added: {track_title} by {artist_name} (item {item_number})")
            else:
                # No exact match found, perform fuzzy matching by track title only
                logger.warning(f"Exact match not found: {track_title} by {artist_name} (item {item_number})")
                
                # Fallback: Search by track title only, and then fuzzy match artist
                params = {
                    'q': f"track:{track_title}",  # Loosen the search by track title only
                    'type': 'track',
                    'limit': 10  # Get more results to allow fuzzy matching of the artist
                }

                response = requests.get(search_url, headers=headers, params=params)
                if response.status_code == 200:
                    tracks = response.json()['tracks']['items']
                    if tracks:
                        # Use fuzzy matching to find the closest artist name
                        best_match = None
                        best_score = 0  # Keep track of the highest similarity score
                        
                        for track in tracks:
                            track_artist = track['artists'][0]['name']
                            track_release_year = track['album']['release_date'].split('-')[0]  # Get the release year

                            # Fuzzy match on artist name and check if the release year is within ±1 of the CSV year
                            similarity_score = fuzz.ratio(artist_name.lower(), track_artist.lower())
                            year_diff = abs(int(track_release_year) - int(track_year))

                            if similarity_score > best_score and year_diff <= 1:  # Year filter of ±1 year
                                best_score = similarity_score
                                best_match = track

                        # If the best match is found and similarity score is above a threshold (e.g., 70), use it
                        if best_match and best_score >= 70:
                            track_uris.append(best_match['uri'])
                            logger.info(f"Fuzzy match added: {track_title} by {best_match['artists'][0]['name']} (item {item_number}, fuzzy match)")
                        else:
                            logger.warning(f"No good fuzzy match on track: {track_title} by {artist_name} (item {item_number})")
                    else:
                        # Fallback: Search by artist name and then fuzzy match track title
                        logger.warning(f"No track results: {track_title} (item {item_number})")

                        params = {
                            'q': f"artist:{artist_name}",  # Now search by artist name only
                            'type': 'track',
                            'limit': 10  # Get more results to allow fuzzy matching of the track title
                        }

                        response = requests.get(search_url, headers=headers, params=params)
                        if response.status_code == 200:
                            tracks = response.json()['tracks']['items']
                            if tracks:
                                # Use fuzzy matching to find the closest track title
                                best_match = None
                                best_score = 0

                                for track in tracks:
                                    track_title_result = track['name']
                                    track_release_year = track['album']['release_date'].split('-')[0]

                                    # Fuzzy match on track title and check if the release year is within ±1 of the CSV year
                                    similarity_score = fuzz.ratio(track_title.lower(), track_title_result.lower())
                                    year_diff = abs(int(track_release_year) - int(track_year))

                                    if similarity_score > best_score and year_diff <= 1:  # Year filter of ±1 year
                                        best_score = similarity_score
                                        best_match = track

                                # If the best match is found and similarity score is above a threshold (e.g., 70), use it
                                if best_match and best_score >= 70:
                                    track_uris.append(best_match['uri'])
                                    logger.info(f"Fuzzy match added by artist: {best_match['name']} by {artist_name} (item {item_number}, fuzzy match)")
                                else:
                                    logger.warning(f"No good fuzzy match on artist: {track_title} by {artist_name} (item {item_number})")
                            else:
                                logger.warning(f"No artist results: {artist_name} (item {item_number})")
                        else:
                            logger.error(f"Error in artist fuzzy search: {response.status_code} (item {item_number})")
                else:
                    logger.error(f"Error in track fuzzy search: {response.status_code} (item {item_number})")
        else:
            logger.error(f"Error in exact search: {response.status_code} (item {item_number})")

        time.sleep(0.5)  # To avoid rate limits

    if track_uris:
        # Add tracks to the playlist in batches of 100
        add_tracks_to_playlist(token, track_uris, playlist_id)

# Main program flow remains the same, where you authenticate and pass token, etc.


# Replace these with your actual values
CLIENT_ID = 'e6fa33c5d4884f39afd576b7deb744d2'
CLIENT_SECRET = '737b5b230f57450aaa6f45b000906840'
REDIRECT_URI = 'http://localhost:888/callback'  # Your redirect URI
SCOPE = 'playlist-modify-public playlist-modify-private'  # Scopes needed for your app

# Spotify API endpoints
AUTH_URL = 'https://accounts.spotify.com/api/token'
API_URL = 'https://api.spotify.com/v1'


# Step 1: Generate the URL to authorize the user
def get_authorization_url(client_id, redirect_uri, scope):
    url = 'https://accounts.spotify.com/authorize'
    params = {
        'client_id': client_id,
        'response_type': 'code',
        'redirect_uri': redirect_uri,
        'scope': scope
    }
    request_url = requests.Request('GET', url, params=params).prepare().url
    return request_url

# Step 2: Exchange the authorization code for access and refresh tokens
def exchange_code_for_tokens(client_id, client_secret, code, redirect_uri):
    token_url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': 'Basic ' + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode(),
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': redirect_uri
    }

    response = requests.post(token_url, headers=headers, data=data)

    if response.status_code == 200:
        tokens = response.json()
        return tokens['access_token'], tokens['refresh_token']  # Return both access and refresh tokens
    else:
        logging.error(f"Error fetching tokens: {response.status_code} - {response.json()}")
        return None, None

# Step 3: Refresh access token using the refresh token
def refresh_access_token(client_id, client_secret, refresh_token):
    token_url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': 'Basic ' + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode(),
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token
    }

    response = requests.post(token_url, headers=headers, data=data)
    
    if response.status_code == 200:
        return response.json()['access_token']
    else:
        logging.error(f"Error refreshing token: {response.status_code} - {response.json()}")
        return None

# Step 4: Check if the playlist already exists
def get_existing_playlist(token, user_id, playlist_name):
    url = f"{API_URL}/users/{user_id}/playlists"
    headers = {
        'Authorization': f'Bearer {token}'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        playlists = response.json()['items']
        for playlist in playlists:
            if playlist['name'] == playlist_name:
                return playlist['id']  # Return the existing playlist ID
    else:
        logging.error(f"Failed to fetch playlists: {response.status_code} - {response.json()}")
    return None

# Step 5: Create a new playlist
def create_playlist(token, user_id, playlist_name):
    existing_playlist_id = get_existing_playlist(token, user_id, playlist_name)
    if existing_playlist_id:
        logging.info(f"Playlist '{playlist_name}' already exists. Using existing playlist.")
        return existing_playlist_id

    url = f"{API_URL}/users/{user_id}/playlists"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = json.dumps({
        'name': playlist_name,
        'description': 'A playlist of 90s songs',
        'public': False
    })

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 201:
        return response.json()['id']
    else:
        logging.error(f"Failed to create playlist: {response.status_code} - {response.json()}")
        return None

# Main program flow
if __name__ == '__main__':
    # Step 1: Get the authorization URL and direct the user to it
    auth_url = get_authorization_url(CLIENT_ID, REDIRECT_URI, SCOPE)
    print(f"Go to the following URL to authorize the app: {auth_url}")

    # Step 2: After user authorization, Spotify will redirect with the "code" query parameter.
    # Example: http://localhost:8888/callback?code=YOUR_CODE
    authorization_code = input("Enter the authorization code from the URL: ")

    # Step 3: Exchange the authorization code for access and refresh tokens
    access_token, refresh_token = exchange_code_for_tokens(CLIENT_ID, CLIENT_SECRET, authorization_code, REDIRECT_URI)

    if access_token and refresh_token:
        logging.info("Access and refresh tokens successfully acquired.")
        print(f"Access Token: {access_token}")
        print(f"Refresh Token: {refresh_token}")

        # Step 4: Now, use the access token to create a playlist and add tracks from the CSV
        USER_ID = '31ikknp5hgwcpwwgnbcwlinfpsyu'  # Replace with your Spotify user ID
        playlist_name = "90s-song-list"
        
        # Create or use the existing playlist
        playlist_id = create_playlist(access_token, USER_ID, playlist_name)

        if playlist_id:
            csv_file_path = '90s_song_list.csv'  # Replace with the path to your CSV file
            search_and_add_tracks_to_playlist(access_token, csv_file_path, playlist_id)
    else:
        logging.error("Failed to acquire tokens.")