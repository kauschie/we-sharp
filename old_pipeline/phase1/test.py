import os
import time
import shutil
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from boxsdk import Client, JWTAuth
from selenium.webdriver.support.ui import Select

def upload_midi_to_box(midi_file_path):
    # Initialize Box client
    auth = JWTAuth.from_settings_file('./keypair.json')
    client = Client(auth)

    # Box folder ID (music/midi-p1/orig)
    box_root_folder_id = '303861755974'

    # Upload the MIDI file
    with open(midi_file_path, 'rb') as file_stream:
        midi_file_name = os.path.basename(midi_file_path)
        print(f"Uploading '{midi_file_name}' to Box...")
        try:
            client.folder(folder_id=box_root_folder_id).upload_stream(file_stream, midi_file_name)
            print(f"'{midi_file_name}' uploaded successfully.")
        except Exception as e:
            if 'item_name_in_use' in str(e):
                # Rename the file and try again
                new_midi_file_name = f"{os.path.splitext(midi_file_name)[0]}_{int(time.time())}.mid"
                new_midi_file_path = os.path.join(os.path.dirname(midi_file_path), new_midi_file_name)
                os.rename(midi_file_path, new_midi_file_path)
                with open(new_midi_file_path, 'rb') as new_file_stream:
                    client.folder(folder_id=box_root_folder_id).upload_stream(new_file_stream, new_midi_file_name)
                    print(f"'{new_midi_file_name}' uploaded successfully.")
            else:
                raise e

def download_midi_file(driver, download_directory, target_directory):
    try:
        try:
            # if the button exists click the keep using for free button
            cancel_button = driver.find_element(By.CLASS_NAME, "full")
            if cancel_button:
                cancel_button.click()
        except:
            pass

        # Wait for and click the generate button
        generate_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(@title, 'Generate a new melody')]"))
        )
        generate_button.click()

        # Wait a moment for generation
        time.sleep(2)

        # Wait for and click the MIDI download button
        download_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(@title, 'Download MIDI file')]"))
        )
        download_button.click()

        # Wait for download to complete
        time.sleep(5)

        try:
            # if the button exists click the keep using for free button
            cancel_button = driver.find_element(By.CLASS_NAME, "full")
            if cancel_button:
                cancel_button.click()
        except:
            pass

        # Find the most recent MIDI file in the download directory
        midi_files = [f for f in os.listdir(download_directory) if f.endswith('.mid')]
        if not midi_files:
            raise Exception("No MIDI file found in downloads")

        # Get the most recently downloaded file
        latest_midi = max([os.path.join(download_directory, f) for f in midi_files], key=os.path.getctime)

        # Move the file to the target directory
        target_path = os.path.join(target_directory, os.path.basename(latest_midi))
        shutil.move(latest_midi, target_path)
        print(f"MIDI file successfully downloaded and moved to {target_path}")
        
        # Upload the MIDI file to Box with retries
        max_retries = 5
        retry_delay = 2  # seconds between retries

        for attempt in range(max_retries):
            try:
                upload_midi_to_box(target_path)
                print(f"Successfully uploaded to Box on attempt {attempt + 1}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:  # Last attempt failed
                    print(f"Failed to upload to Box after {max_retries} attempts")
                    print(f"Final error: {e}")
            
            # Delete the MIDI file from the target directory
            os.remove(target_path)
        
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Record the start time
    start_time = time.time()

    # Directories
    download_dir = os.path.expanduser("~/Downloads")
    target_dir = r"C:/Users/jst1b/source/repos/we-sharp/we-sharp/phase1/midi" # Replace with your target directory

    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Set up Chrome options
    chrome_options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": os.path.abspath(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "safebrowsing.disable_download_protection": True,
        "profile.default_content_setting_values.automatic_downloads": 1 
    }
    chrome_options.add_experimental_option("prefs", prefs)

    # Initialize the Chrome WebDriver
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Navigate to the website
        driver.get("https://dopeloop.ai/melody-generator/") 
        
        # Wait for and click edit button
        edit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@title='Edit melody settings']"))
        )
        edit_button.click()

        # Wait for modal to open and find length input
        length_selector= WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//select[@title='Loop length']"))
        )
        length_selector.click()
        
        # Find the select element
        length_select = Select(driver.find_element(By.NAME, "loop-length"))

        # Select by visible text
        length_select.select_by_visible_text("64")
        
        time.sleep(1)

        # Infinite loop to generate, download, and upload MIDI files
        while True:
            download_midi_file(driver, download_dir, target_dir)
            # Delete the MIDI file
            if os.path.exists(target_dir):
                # Loop through all files and subdirectories
                for filename in os.listdir(target_dir):
                    file_path = os.path.join(target_dir, filename)
                    
                    # If it's a file, remove it
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the browser
        driver.quit()

        # Record the end time
        end_time = time.time()

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()