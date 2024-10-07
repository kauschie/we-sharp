import subprocess
import logging
import time

# Function to get the current IP address
def get_current_ip():
    """Returns the current external IP address."""
    try:
        result = subprocess.run(['curl', '-s', 'ifconfig.me'], stdout=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except Exception as e:
        logging.error(f"Error fetching current IP: {e}")
        return None

def ensure_vpn_connection():
    """Ensure the VPN connection is fully established before downloading."""
    old_ip = get_current_ip()
    if not old_ip:
        logging.error("Could not retrieve initial IP address.")
        return False

    logging.info(f"Initial IP: {old_ip}")

    # Switch PIA server
    logging.info("Switching PIA server...")
    subprocess.run(['python3', 'change_pia_server.py'], check=True)

    # Wait for the IP to change
    retries = 10
    delay = 5
    for attempt in range(retries):
        new_ip = get_current_ip()
        if new_ip and new_ip != old_ip:
            logging.info(f"VPN connected successfully! New IP: {new_ip}")
            return True
        logging.info(f"Attempt {attempt + 1}/{retries}: VPN not connected yet. Current IP: {new_ip}")
        time.sleep(delay)

    logging.error("Failed to establish VPN connection after retries.")
    return False

# Main function to download videos
def download_video(video_url):
    """Download video using yt-dlp after ensuring VPN connection."""
    logging.info(f"Downloading audio for {video_url}")
    
    # Ensure VPN connection before download
    if ensure_vpn_connection():
        subprocess.run(['yt-dlp', '-x', '--audio-format', 'mp3', video_url], check=True)
        logging.info(f"Download of {video_url} completed successfully.")
    else:
        logging.error("VPN connection failed. Skipping download.")

# Example usage: list of video URLs to download
video_urls = [
    'https://www.youtube.com/watch?v=kJQP7kiw5Fk',
    'https://www.youtube.com/watch?v=BvJSig2WhnY'
]

for video_url in video_urls:
    download_video(video_url)
