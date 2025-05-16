#!/usr/bin/env python
"""
Audio Hash Comparison Tool

Compares audio files by hashing their content and detects duplicates 
between a probe directory and a dataset directory.
"""

import os
import hashlib
import wave
import json
import csv
import numpy as np
import logging
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt


def setup_logging(log_file='audio_hash_comparison.log'):
    """Set up logging to both console and file."""
    logger = logging.getLogger('AudioHashComparator')
    logger.setLevel(logging.INFO)
    
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def calculate_wav_hash(file_path, method='content'):
    """Calculate a hash for WAV files using specified method."""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            params = wav_file.getparams()
            
            if method == 'metadata':
                metadata_str = (
                    f"{params.nchannels}|"
                    f"{params.sampwidth}|"
                    f"{params.framerate}|"
                    f"{params.nframes}"
                )
                return hashlib.md5(metadata_str.encode()).hexdigest()
            
            elif method == 'content':
                wav_file.rewind()
                content = wav_file.readframes(params.nframes)
                return hashlib.sha256(content).hexdigest()
            
            elif method == 'comprehensive':
                wav_file.rewind()
                frames = wav_file.readframes(params.nframes)
                audio_array = np.frombuffer(frames, dtype=np.int16)
                samples = audio_array[::len(audio_array)//10]
                
                metadata_str = (
                    f"{params.nchannels}|"
                    f"{params.sampwidth}|"
                    f"{params.framerate}|"
                    f"{params.nframes}"
                )
                
                hasher = hashlib.sha256()
                hasher.update(metadata_str.encode())
                hasher.update(samples.tobytes())
                
                return hasher.hexdigest()
            
            else:
                raise ValueError("Invalid hashing method")
    
    except Exception as e:
        print(f"Error hashing file {file_path}: {e}")
        return None


def hash_directory(directory_path, logger, method='content'):
    """Generate hashes for all WAV files in a directory."""
    file_hashes = {}
    wav_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.wav')]
    
    logger.info(f"Hashing {len(wav_files)} WAV files in {directory_path}")
    
    for filename in tqdm(wav_files, desc="Hashing Files", unit="file"):
        file_path = os.path.join(directory_path, filename)
        file_hash = calculate_wav_hash(file_path, method)
        
        if file_hash:
            file_hashes[filename] = file_hash
    
    return file_hashes


def compare_probe_to_dataset(dataset_directory, probe_directory, logger, method='content'):
    """Compare probe files against a dataset of audio files."""
    dataset_hashes = hash_directory(dataset_directory, logger, method)
    logger.info(f"Hashed {len(dataset_hashes)} files in the dataset.")
    
    probe_matches = {}
    comparison_output = []
    probe_files = [f for f in os.listdir(probe_directory) if f.lower().endswith('.wav')]
    
    logger.info(f"Comparing {len(probe_files)} probe files against dataset")
    
    for probe_filename in tqdm(probe_files, desc="Comparing Probe Files", unit="file"):
        probe_path = os.path.join(probe_directory, probe_filename)
        probe_hash = calculate_wav_hash(probe_path, method)
        
        if not probe_hash:
            logger.warning(f"Could not hash probe file: {probe_filename}")
            continue
        
        matches = [
            dataset_name for dataset_name, dataset_hash in dataset_hashes.items() 
            if dataset_hash == probe_hash
        ]
        
        comparison_record = {
            'probe_filename': probe_filename,
            'probe_hash': probe_hash,
            'is_match': bool(matches),
            'matched_dataset_files': ', '.join(matches) if matches else 'No match'
        }
        comparison_output.append(comparison_record)
        
        if matches:
            logger.info(f"Probe file {probe_filename} matches dataset files: {matches}")
            probe_matches[probe_filename] = matches
        else:
            logger.info(f"Probe file {probe_filename} is unique (no match in dataset).")
    
    csv_output_path = 'hash_comparison_detailed.csv'
    csv_keys = ['probe_filename', 'probe_hash', 'is_match', 'matched_dataset_files']
    
    with open(csv_output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_keys)
        writer.writeheader()
        writer.writerows(comparison_output)
    
    logger.info(f"Detailed hash comparison written to {csv_output_path}")
    
    return probe_matches


def visualize_matches(probe_matches, logger):
    """Create visualizations of match results."""
    if not probe_matches:
        logger.warning("No probe matches to visualize.")
        return
    
    matched_files = len([m for m in probe_matches.values() if m])
    unique_files = len(probe_matches) - matched_files
    
    if matched_files > 0 or unique_files > 0:
        plt.figure(figsize=(10, 6))
        plt.pie(
            [matched_files, unique_files], 
            labels=['Matched Files', 'Unique Files'], 
            autopct='%1.1f%%',
            colors=['#ff9999','#66b3ff']
        )
        plt.title('Probe Files: Match Distribution')
        plt.savefig('match_distribution.png')
        plt.close()
    
    match_counts = {}
    for matches in probe_matches.values():
        for match in matches:
            match_counts[match] = match_counts.get(match, 0) + 1
    
    if match_counts:
        plt.figure(figsize=(12, 6))
        plt.bar(match_counts.keys(), match_counts.values())
        plt.title('Frequency of Dataset File Matches')
        plt.xlabel('Dataset Files')
        plt.ylabel('Number of Matches')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('match_frequency.png')
        plt.close()


def main():
    logger = setup_logging()
    
    try:
        dataset_directory = "" # e.g., "./data/audio_lm/eval"
        probe_directory = "" # e.g., "./data/audio_lm/probe"
        
        probe_matches = compare_probe_to_dataset(
            dataset_directory, 
            probe_directory, 
            logger
        )
        
        visualize_matches(probe_matches, logger)
        
        with open('probe_matches.json', 'w') as f:
            json.dump(probe_matches, f, indent=4)
        
        logger.info("Comparison process completed successfully.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()