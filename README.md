# we-sharp
Senior Project Repo

Authors

Michael Kausch  
Huaiyu Zhang  
Dominic Flores  
Braden Stitt

Operation Manual

We-Sharp

# Table of Contents

This manual explains the different portions of the project and needed steps for reproduction. Below are the individual sections necessary to complete for full operation of the GUI which is covered in the final section of the manual. To fully reproduce the project from start to finish, sections 1 through 3 can be completed in any order, however, section one goes over where to clone the main repository before installing the AudioLM portion. If the section describes methodology to train on their own data, then the final subsection will contain the path to download the full checkpoints used during the 2025 Senior Project expo. Otherwise, it assumes the end user wishes to train from start to finish.

# AUDIOLM SETUP

## SETUP

### SYSTEM REQUIREMENTS

Training makes use of CUDA technology, if you wish to manually train then you will need a GPU with associated drivers enabling CUDA 12.2xx. There are additional dependencies which are addressed in the below walkthrough:

### MANUAL INSTALLATION

Install miniconda on linux:

wget <https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh>

bash Miniconda3-latest-Linux-x86_64.sh

Ensure that you have the following dependencies installed:

build-essential

ffmpeg

sox

boxsdk

fluidsynth

Follow the steps and restart your terminal, or execute:

source ~/.bashrc

Pull the project package from github: <https://github.com/kauschie/we-sharp>

git clone <https://github.com/kauschie/we-sharp>

The environment.yml file should already assist with downgrading pip, but as an extra precaution it’s advised to downgrade pip to a previous version before creating the environment as well due to a bug in facebook’s fairseq library that affects audiolm’s installation process and dependencies:

conda install -y "pip<24.1"

Then, use the environment file located in we-sharp/audiolm/setup to create a conda environment:

conda env create -f we-sharp/audiolm/environment.yml

conda activate wesharp_env

Then install audiolm_pytorch library without dependencies:

pip install audiolm-pytorch --no-deps

### LIBRARY MODIFICATION

The original library was modified to incorporate new functionality for our project so the original library’s files need to be modified. Copy the contents of the audiolm/lib_mods/ directory to the audiolm_pytorch library.

Example command if installed using miniconda3 to this directory:

cp ./audiolm/lib_mods/\*.py /miniconda3/envs/audiolm_env/lib/python3.10/site-packages/audiolm_pytorch/.

## ACQUISITION OF TRAINING DATA

### USING PRETRAINED MODEL CHECKPOINTS

If you wish to avoid training your own model, you can download the model checkpoints found here: <https://csub.app.box.com/folder/321403914865>

The Hubert models should be downloaded to /audiolm/models/ while the checkpoints should be downloaded to /audiolm/results/. This requires the least amount of reconfiguration from the filebase.

### DOWNLOAD OUR DATA

<https://csub.app.box.com/folder/303861755974> (original midi files)

<https://csub.app.box.com/folder/305425545289> (wav files already converted from midi)

Note, this data still needs to pre-processed which can be followed below.

### CUSTOM DATASET

If using your own dataset, it’s assumed at this point that all files are either in MIDI form (.mid or .midi file extension) or WAV form.

### PREPROCESSING DATASET

If your input files are in .midi or .mid format, they must be converted to .wav before training. This can be done using the pp_midi.py script.

Before running the script, install the required dependencies:

\# Install python packages

pip install soundfile pretty_midi

\# Install system dependency

sudo apt install fluidsynth

Fluidsynth is required to render MIDI files into audio using a SoundFont (.sf2) file. The following variables need to be changed in the pp_midi.py script:

&nbsp;   input_directory = os.path.join(base, "hz_midi")

&nbsp;   output_directory = os.path.join(base, "hz_16k_wav")

&nbsp;   completed_directory = os.path.join(base, "hz_processed_midi")

&nbsp;   baddies_directory = os.path.join(base, "baddies")

&nbsp;   soundfont = "Timbres of Heaven (XGM) 4.00(G).sf2"

Note, the soundfount is available from the project files Box cloud folder located here (<https://csub.app.box.com/folder/321403914865>)

This script will convert all files to 16khz wavs if it detects and malformed midi files or midi files without piano tracks, it will remove them.

The files then need to be processed into both 16kHz and 24kHz files. The 16kHz files will be used to train the semantic transformer because wav2vec expects this format as it’s input. The 24kHz files are needed by EnCodec and will be used in the coarse/fine transformers. Note, you should make sure that that the data paths for the training files are appropriate and accurate or it the resultant output will be very sped up as a result. If you used midis from above, they should already be 16kHz so it will execute very quickly.

#### Preprocess and segment wav files

In midi_cut_songs.py, modify as needed:

\# Parameters

segment_length_ms = 2000  # 2 seconds, adjust to the length of choice

rms_threshold = 100  # Adjust this if needed after testing

\# (typical range 50–200)

\# Paths

input_folder = "hz_16k_wav"

output_folder_16k = "hz_2s_16k"

output_folder_24k = "hz_2s_24k"

log_file = "blank_segments_log.txt"

### SIMPLEST CASE DATASET FOR TESTING THE TRAINING PIPELINE

If testing the training pipeline from end to end to ensure that everything is working, you can create 100 copies of one file using the copy_song.py script found in /we-sharp/audiolm/fixes/copy_song.py script. Just change the location of the file location at the top where the \`original_file\` variable is and this script will create 100 copies of the same song to a directory for you to use to test and overfit the model to. Playing back the output from the overtrained model should result in the same file the model was trained on to ensure that all pieces are functioning correctly.

## TRAINING

If you downloaded the pre-trained checkpoints, you can skip this section.

### MODIFY SCRIPT WITH SYSTEM SPECIFICS

The scripts will need to be updated for your particular directory layout as it was previously hard-coded to work across users on Dr. Cruz’s training server, Athena.

The following variables will need to be modified with their location on your system:

log_dir = './logs/sem'

hubert_checkpoint_path = './models/hubert_base_ls960.pt'

hubert_kmeans_path = './models/hubert_base_ls960_L9_km500.bin'

dataset_path = "/home/mkausch/dev/audiolm/hz_2s_16k"

results_folder = './results'  # Results directory

while optionally altering the following:

training_max = 100000 # final training step before script finishes

model_save = 1000 # saves every 1000 steps, adjust to your liking

results_save = 1000 # used as an interval to calculate validation loss

temp_max_length = 16000\*2 #(should be a multiple of the sample rate)

The hyperparameters used for the transformers for this project should already be configured within each training script.

### EXECUTE THE SPECIFIC TRAINING SCRIPT

After setting up the training script to have all of the correct directory locations for the training, data files and checkpoint paths, the user can launch the training script with python

Ex)

python semantic_train.py

python coarse_train.py

python fine_train.py

The loss should be steadily below 1.0 to produce music as heard during the demonstration. See below for information on viewing training curves and logs.

### STARTING/STOPPING TRAINING

Training will come to an end once it reaches the maximum step set by the global variable

_training_max._ If the user desires to exit training before the predefined value, they can issue a sigint signal interrupt using the keyboard shortcut _ctrl + c_ in the terminal window and a message will prompt asking the user if they wish to save the training progress to

### TRAINING LOGS

#### Manual Logs

All hyperparameters are output to a log file in the audiolm/logs/ directory with the logs from semantic training appearing in the sem/ subdirectory in the semantic_training.log, coarse training at coarse/coarse_training.log and fine training logs found at fine/fine_training.log. Additionally, all loss values, validation set values and model saves are also logged in their respective training log.

#### Tensorboard

If you installed tensorboard, the training logs can be viewed by executing:

tensorboard --logdir=”path/to/the/log/director/of/your/choice”

and then using your browser of choice to navigate to:

<http://localhost:6006>

and the logs from the training session should be visible using Tensorboard’s interface.

## INFERENCE / TESTING

### ENSURE CORRECT PARAMETERS IN SCRIPT

Describe how the hyperparameters need to be updated in the gen_audio_batch2.py script. They can be found in the semantic_train.py, coarse_train.py, and fine_train.py scripts or their associated log files.

Update these paths with the real checkpoints:

sem_path = "./path/to/checkpoint/semantic.transformer.25000.pt"

coarse_path = "./path/to/checkpoint/coarse.transformer.29219.pt"

fine_path = "./path/to/checkpoint/fine.transformer.24245.pt"

hubert_checkpoint_path = "./models/hubert_base_ls960.pt"

hubert_kmeans_path = "./models/hubert_base_ls960_L9_km500.bin"  

If you are not training your own checkpoints, all model checkpoints are available at the following folder using Box cloud storage:

<https://csub.app.box.com/folder/321403914865>

Update the following parameters for each of the transformers to the matching values from training. You will need to update these if you modified them previously, otherwise they should be the same as the original training.

semantic_transformer = SemanticTransformer(

&nbsp;   num_semantic_tokens=wav2vec.codebook_size,

&nbsp;   dim=1024,

&nbsp;   depth=12,

&nbsp;   heads=16

&nbsp;   ).cuda()

semantic_transformer.load(sem_path)

coarse_transformer = CoarseTransformer(

&nbsp;   num_semantic_tokens=wav2vec.codebook_size,

&nbsp;   codebook_size=1024,

&nbsp;   num_coarse_quantizers=3,

&nbsp;   dim=1024,

&nbsp;   depth=6,

&nbsp;   heads=16

&nbsp;   ).cuda()

coarse_transformer.load(coarse_path)

fine_transformer = FineTransformer(

&nbsp;   num_coarse_quantizers=3,

&nbsp;   num_fine_quantizers=5,

&nbsp;   codebook_size=1024,

&nbsp;   dim=1024,

&nbsp;   depth=6,

&nbsp;   heads=16

&nbsp;   ).cuda()

fine_transformer.load(fine_path)

If any discrepancies arise with the configuration of the transformers, the program will not launch. Verify that you have everything setup correctly with your training scripts and/or log files.

### MANUAL EXECUTION OF GEN_AUDIO_BATCH2.PY

Change into the src directory

cd src/

The script that performs inference can then be executed as follows:

python gen_audio_batch2.py --duration 8 --batch_size 4 --prime_wave seed_files/myseed.wav --output my_generated_track

This will generate 4 audio clips called my_generated_track1.wav, my_generated_track2.wav, my_generated_track3.wav, and my_generated_track4.wav, all of which are at least 8 seconds in length and it will use a prime/seed file taken from the audio file seed_files/myseed.wav.

# EVALUATION METRICS SETUP

## QUICK USE

To rigorously evaluate the performance of the AudioLM pipeline, we implement two quantitative metrics: **Audio Hash Comparison** and **Frechet Audio Distance (FAD)**. Each tool serves a distinct purpose:

- **Audio Hashing**: Identifies exact or near-duplicate generations (memorization detection).
- **FAD**: Measures perceptual similarity between real and generated audio, assessing output quality.

These metrics are used post-inference and are agnostic to model architecture.

## Audio Hash Comparison Tool

**Purpose:**

To detect whether generated audio is directly memorized from the training set or is unintentionally duplicated. This ensures the model is generalizing rather than overfitting to training data.

**How it works:**

SHA-256 hashes are computed for:

- A reference dataset (e.g., training/validation data)
- A probe set (e.g., generated audio outputs)

Matching hashes between probe and dataset files indicate identical audio files.

**Setup and Execution:**

- Open the audio_hash.py script.
- Modify these lines at the top to point to your directories:

**dataset_directory = "/absolute/path/to/reference/dataset"**

**probe_directory = "/absolute/path/to/generated/audio"**  

**Run the script:**

Python audio_hash.py

**Output:**

- hash_comparison_detailed.csv: Tabulated match results (filename, match found, hash values).
- match_distribution.png: Histogram of duplicate matches across probes.
- match_frequency.png: Heatmap of match frequency by dataset region (useful if dataset is segmented).

**Interpretation:**

- There should be no exact hash matches.
- Matches often indicate overfitting.

## Audio Hash Comparison Tool

**Purpose:**  
To evaluate the **perceptual quality** of generated audio by measuring statistical differences in high-level embeddings between real and synthetic audio. Analogous to Frechet Inception Distance (FID) in vision models.

**How it works:**

1. Both real and generated audio files are passed through a pretrained audio embedding model (e.g., CLAP).
2. Mean and covariance statistics of these embeddings are computed for each distribution.
3. The Frechet distance is computed between the two distributions:
    1. Lower FAD = closer the generated distribution is to the real one.

**Setup:**

1. Create the conda environment used for FAD evaluation:

conda env create -f fadtk_environment.yml

conda activate fadtk_env

1. Prepare two directories:

- ./baseline/: Reference real audio samples (e.g., validation data).
- ./eval/: Generated audio outputs.

1. Run the FAD scipt:

python fad.py --baseline ./baseline --eval ./eval --model clap-laion-audio

**Optional flags:**

- \--workers: Set number of data loader workers (default = 8).
- \--use-inf: Enables an approximation using inverse FAD (faster, slightly less precise).
- \--batch-size: Controls embedding model batch size.

**Output:**

- A numeric FAD score printed to stdout and optionally saved to a file if desired.
- Optional log output with sample counts and any errors in loading files.

**Interpretation:**

- FAD ≈ 0: Generated audio is perceptually indistinguishable from real audio.
- FAD > 10: Significant distributional gap; model may be generating unnatural or incoherent outputs.

**Best Practice:**

- Run FAD multiple times across different prompt sets and seed variations.
- Use the same audio preprocessing pipeline for both real and generated files.

# MidiLM DataPreprocessing & LLM Training

## Quick Use

\--Python3 run.py

## Data preprocessing

### Stan_Midi.py

This script converts .mid files into plain text by extracting only note_on and note_off events. It is designed for symbolic music analysis and preprocessing tasks, such as MIDI-based language modeling.

To use the script, place your MIDI files in the TestMidi folder, or modify the INPUT_FOLDER path to match your own directory. When you run the script, it will process each .mid file and output a corresponding .txt file to the MIDI_TXT folder, which will be created automatically if it does not exist.

Each line in the output file represents a single MIDI event in the format: This format is compact and easy to tokenize or transform for use in machine learning pipelines.

The script also collects basic statistics, including the frequency of note pitches, velocity values, and the total number of MIDI events processed. These statistics are printed to the console upon completion.

Error handling is included to ensure that malformed MIDI messages do not interrupt the batch processing. The script skips problematic messages and logs a warning for each one.

This script requires the mido and tqdm Python libraries. You can install them using pip install mido tqdm if they are not already available in your environment.

### full_midi_pipline.py

This script performs a four-step preprocessing pipeline on symbolic music data extracted from MIDI files. It is designed to prepare clean and structured input for downstream tasks such as music generation using language models.

In Step 1, it reads each MIDI-converted .txt file and identifies sequences of simultaneous note events (i.e., events with zero time offset). These blocks are sorted by pitch and rewritten with only the first event retaining its original offset. This standardizes the structure of chord-like clusters.

In Step 2, the script creates a mapping between each note_on and its corresponding note_off event. It scans line-by-line and records note duration information in a separate CSV file for each input.

Step 3 identifies and removes duplicate or conflicting note_on events that share the same pitch but are not followed by proper note_off pairs. It corrects timing offsets and logs the deleted lines to avoid structural inconsistencies during training.

In Step 4, it removes full chord structures by analyzing groups of three or more note_on events with the same duration and time offset. These are considered harmonic blocks and are excluded to isolate the monophonic melody line. The resulting files are saved as melody-only text representations, and statistics about the removed chords are collected in a CSV file.

The entire process is multi-threaded using a thread pool and includes detailed progress bars and log outputs. This script is an essential part of the data preparation pipeline for symbolic music modeling with MIDI-LM or similar token-based architectures.

### Midi_Tokenform.py

This script performs tokenization and sliding window segmentation on preprocessed melody-only MIDI text files. It is intended to prepare training data for language models such as GPT-style architectures applied to symbolic music.

For each MIDI .txt file in the input folder, the script first converts it into three pitch-shifted token versions using offsets of 0, -1, and +1. Each token is formatted as \[on/off\]\_\[pitch\]\_\[delta_time\], where pitch is clamped between 21 and 109 to ensure compatibility with general-purpose MIDI representations.

After tokenization, the script segments each token sequence into overlapping sliding windows of fixed size (1024 tokens) with a stride of 512. If the remaining tokens at the end of a file are longer than a defined minimum (796), the last window is also saved. All resulting windowed segments are written as .txt files to the output directory.

The entire pipeline supports multithreading and uses a thread pool to accelerate processing across files. Progress is visualized using a tqdm progress bar, and the final output reports the total number of sliding window segments generated.

This tokenization and segmentation step is critical in preparing symbolic music data for Transformer-based models, enabling consistent context lengths and data augmentation through pitch shifting.

## LLM Training

### Zhengli.py

This script converts preprocessed symbolic music token sequences into a JSONL dataset format, which is suitable for training autoregressive language models such as GPT-2.

It scans all .txt files in a given folder and selects those that match a specific naming convention, where the filename contains a batch index in the middle (e.g., xxx_0_xxx.txt). Only files with batch number "0" are processed to reduce redundancy or to control dataset size during experimentation.

Each file is read and tokenized using a preloaded Hugging Face PreTrainedTokenizerFast tokenizer. The script expects the resulting token length to be exactly 1024 tokens per file. Files that do not meet this requirement are skipped and reported in the console output.

For each valid file, the tokenized output is written as a JSON object in the format {"input_ids": \[...\]} to a line in a .jsonl file. This format is widely used in Hugging Face Trainer pipelines and ensures compatibility with token-based language model training workflows.

The script also includes progress logging and basic error handling, and automatically creates the output directory if it doesn't exist. This step finalizes the training data preparation by packaging token sequences into a standardized format for model consumption.

### Main.py

This script handles the final stage of the MIDI-LM training pipeline by loading tokenized music data and training a GPT-style language model on it.

The process begins by loading a custom tokenizer from a local JSON file using PreTrainedTokenizerFast. The script then reads tokenized input from a .jsonl file where each line contains a JSON object with an "input_ids" field representing a sequence of 1024 music tokens. If the field is stored as a string, it is converted into a list of integers. The "input_ids" are also duplicated into a "labels" field for next-token prediction training, following the standard causal language modeling setup.

The data is converted into a Hugging Face Dataset object, which is compatible with the Trainer API and can be used directly in PyTorch-based training routines.

Next, the model is initialized via a custom utility function initialize_model(), which sets up a small GPT-2 model (e.g., 12 layers, 12 heads, 768 hidden units). If the tokenizer does not include a padding token, one is added—either by using the existing end-of-sequence token or by defining a new \[PAD\] token. The model’s embedding layer is resized accordingly to reflect the updated vocabulary.

Finally, the model is trained using train_model(), which handles training loop, logging, and saving. The training log file path is printed at the end for reference.

This script is designed to be the last step of an efficient music modeling pipeline, enabling symbolic sequence learning from structured MIDI-derived input.

# Client and Server Setup

## QUICK USE

The client and server are already hosted on Athena and the application can easily be accessed using the following link: <https://athena.cs.csubak.edu/dom/>

For implementing the client and server locally, follow the instructions in the next sections. Copies of code sections or directories are so that our group can demonstrate our production environment while also providing a dev environment for implementing the client and server locally.

## Requirements

1. Conda must be installed to use the **_client_env_** virtual environment.
2. Inside the **_gui_** directory, create a directory called **_audio_**.
3. Inside **_audio_**, create two directories: **_uploaded_** and **_generated_**
4. Generate SSL Certificates in the **_client_dev_** (not client) directory
    1. openssl genrsa -out server.key 2048
    2. openssl req -new -key server.key -out server.csr
    3. openssl x509 -req -in server.csr -signkey server.key -out server.crt -days 365

## Server Setup

- Make sure that all requirement steps are completed
- Open a terminal and go to the **_server_** directory (Athena)
- Activate the virtual environment using this command:
  - **_conda activate client_env_**
- Use this command to run the client on your local device:
  - **_npm run dev_**
  - Change the port number in dev_server.js if it is being used
- Click the Server URL
- A security prompt will appear
  - Hit Advanced or Advanced Options
  - Hit Proceed

## Client Setup

- Make sure that **_server setup_** steps have been completed
- Open a terminal and go to the **_client_dev_** (not client) directory (Athena)
- Activate the virtual environment using this command:
  - **_conda activate client_env_**
- Use this command to run the client on your local device:
  - **_npm run dev_**
- Click the URL that is next to **_Network_**. The local host option does not work.
- A security prompt will appear:
  - Hit Advanced or Advanced Options
  - Hit Proceed
