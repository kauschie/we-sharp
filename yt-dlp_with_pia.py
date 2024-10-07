import yt_dlp
import os
import subprocess
import time
import logging
import argparse

# Set up logging
logging.basicConfig(
    filename='yt_dlp_downloads.log', 
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def get_current_ip():
    """Returns the current external IP using curl ifconfig.me."""
    try:
        result = subprocess.run(['curl', '-s', 'ifconfig.me'], stdout=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except Exception as e:
        logging.error(f"Error getting current IP: {e}")
        return None

def download_audio(url, use_proxy=False, proxy_url=None, output_dir='downloads'):
    """Downloads only the audio from a YouTube video or playlist URL."""
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # yt-dlp options for downloading audio
    ydl_opts = {
        'format': 'bestaudio/best',  # Get the best quality audio
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',  # Use FFmpeg to extract audio
            'preferredcodec': 'mp3',  # Convert to mp3
            'preferredquality': '192',  # Set the quality of the audio
        }],
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',  # Set the download directory and file name
        'noplaylist': True,  # Prevent playlist download if single video URL
        'limit-rate': '500K',  # Limit download speed to 500KB/s
        'sleep-interval': 10,  # Minimum sleep interval between requests
        'max-sleep-interval': 60,  # Maximum sleep interval between requests
        'concurrent-fragments': 1,  # Download one fragment at a time
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',  # Spoof user-agent
        'cookies': 'cookies.txt',  # Path to the clean cookies.txt
        'verbose': True  # Enable verbose mode to capture more detailed logs
    }

    # If proxy usage is enabled, add proxy configuration
    if use_proxy and proxy_url:
        ydl_opts['proxy'] = proxy_url  # Set the proxy URL
        logging.info(f"Using proxy: {proxy_url}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logging.info(f"Download of {url} completed successfully.")
    except yt_dlp.utils.DownloadError as e:
        if "captcha" in str(e).lower():
            logging.warning("CAPTCHA detected! Solve the CAPTCHA before retrying.")
        else:
            logging.error(f"Error downloading {url}: {e}")

def download_audio_from_file(file_path, use_proxy=False, proxy_url=None, output_dir='downloads'):
    """Reads a text file containing URLs and downloads audio for each URL."""
    
    if not os.path.isfile(file_path):
        logging.error(f"The file {file_path} does not exist.")
        return
    
    with open(file_path, 'r') as file:
        urls = file.readlines()
        
        for url in urls:
            url = url.strip()
            if url:
                logging.info(f"Downloading audio for {url}")
                download_audio(url, use_proxy, proxy_url, output_dir)

if __name__ == "__main__":
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Download YouTube audio with optional proxy usage.")
    parser.add_argument('--use-proxy', action='store_true', help='Enable proxy before downloading')
    parser.add_argument('--proxy-url', type=str, help='Proxy URL to route traffic through, e.g., http://username:password@proxy_ip:proxy_port')
    parser.add_argument('--file', type=str, default="song_list.txt", help='File containing URLs to download audio from')
    parser.add_argument('--output-dir', type=str, default='downloads', help='Directory to save the downloaded files')
    args = parser.parse_args()

    # Example usage:
    download_audio_from_file(args.file, args.use_proxy, args.proxy_url, args.output_dir)
