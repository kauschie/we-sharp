# Usage Examples:
# 
#   NOTE:   Datafiles should be preprocessed to 24kHz and Mono
#           Though, Demucs performs better with Stereo (2ch) so multiple
#               preprocessing steps are needed
#

# python3 encodec_demo.py --encode-to-discrete faded_low_mono_trim.wav assets/tokens
#       --> encodes wav file "faded_low_mono_trim.wav" and outputs codebooks 
#               to the assets dir as tokens_{bandwidth}.txt where
#               bandwidth is the tested bandwidth size on the encodec model
#               NOTE: The test should be removed when you go to use it 

# python3 encodec_demo.py --decode-from-discrete tokens.txt reconstruct.wav
#       --> decodes contents of tokens.txt into a wav file
#       --> Can easily count files with the following terminal command:
#   cat assets/tokens_1.5.txt | tr -d "[" | tr -d "]" | tr "," "\n" | grep -v '^$' | wc -l  
#   But, N_tokens = (bandwidth) * 1000
#       So, Bandwidth of 1.5 produces 2 codebooks of 775 each or 1500 total tokens

#
# python encodec_demo.py faded_low_mono_trim.wav assets   
#       --> directly encodes then decodes file <faded_low_mono_trim.wav> as 
#               faded_low_mono_trim_{bandwidth}.wav into the assets/ directory
#               for quality comparisons
#               NOTE: this should be removed and only one bandwidth chosen


import argparse
import torchaudio
from transformers import EncodecModel
import torch

# Function to normalize audio to [-1.0, 1.0]
def normalize_audio(audio):
    max_val = torch.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio  # If the signal is silent (max_val = 0), return unchanged

def encode_to_discrete(input_file, output_file, model, device, b=12):
    print(f"Encoding {input_file} to discrete representation...")
    wav, sr = torchaudio.load(input_file)

    # Debug: Check audio shape and sampling rate
    print(f"Loaded audio shape: {wav.shape}, Sample rate: {sr}")

    # Resample to 24 kHz if necessary
    target_sr = 24000
    if sr != target_sr:
        print(f"Resampling from {sr} Hz to {target_sr} Hz...")
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)

    # Normalize the audio
    wav = normalize_audio(wav)

    # Move to the appropriate device
    wav = wav.unsqueeze(0).to(device)

    # Debug: Confirm shape after resampling
    print(f"Audio shape after resampling: {wav.shape}")

    # Encode to discrete representation
    with torch.no_grad():
        encoded = model.encode(wav, bandwidth=b)

    # Debug: Inspect encoded structure
    print(f"Encoded audio codes: {encoded['audio_codes']}")

    # Save discrete representation to a text file
    with open(f"{output_file}_{b}.txt", "w") as f:
        for i, codebook in enumerate(encoded["audio_codes"]):
            # Debug: Print the codebook to check its contents
            print(f"Codebook {i}: {codebook}")

            # Flatten nested list structure and write each sublist to its own line
            for j, code in enumerate(codebook.squeeze(0).cpu().tolist()):
                # Debug: Print each individual code
                print(f"Codebook {i}, Code {j}: {code}")

                # Write the list as a string
                f.write(str(code) + "\n")

            # Add a blank line between codebooks for readability
            f.write("\n")

    print(f"Discrete representation saved to {output_file}.")
def decode_from_discrete(input_file, output_file, model, device):
    print(f"Decoding discrete representation from {input_file} to audio...")

    # Load discrete representation from text file
    with open(input_file, "r") as f:
        blocks = f.read().strip().split("\n\n")  # Split by blank lines between codebooks

        # Debug: Check how many codebooks were found
        print(f"Number of codebooks found: {len(blocks)}")

        discrete_repr = []
        for i, block in enumerate(blocks):
            # Debug: Print each block to verify contents
            print(f"Block {i}: {block.strip()}")

            codes = []
            for line in block.strip().split("\n"):
                # Debug: Print each line before conversion
                print(f"Block {i}, Line: {line}")

                # Convert line (stringified list) back into a list of integers
                code = eval(line)  # Safely evaluate the string to recover the list
                codes.append(code)

            # Convert the list of lists into a tensor
            discrete_repr.append(torch.tensor(codes, device=device))

    # Debug: Confirm the shape of the decoded representation
    print(f"Decoded representation shape: {[repr.shape for repr in discrete_repr]}")

    # Reshape into the format expected by the model
    encoded = {"audio_codes": [repr.unsqueeze(0) for repr in discrete_repr]}

    # Generate audio_scales with the correct shape
    audio_scales = [
        torch.ones(codebook.shape[-1], device=device) for codebook in encoded["audio_codes"]
    ]

    # Debug: Check audio_scales
    print(f"Generated audio_scales: {audio_scales}")

    # Decode to audio
    with torch.no_grad():
        decoded = model.decode(encoded["audio_codes"], audio_scales)["audio_values"].cpu()

    # Reshape decoded tensor to match `[num_channels, num_samples]`
    if decoded.ndim == 3:  # Shape is `[750, 1, num_samples]`
        decoded = decoded.squeeze(1)  # Remove the singleton dimension

    if decoded.ndim == 2:  # Shape is `[750, num_samples]`
        decoded = decoded.mean(dim=0, keepdim=True)  # Mix 750 channels into 1 (mono)

    # Debug: Confirm the shape of the tensor to be saved
    print(f"Decoded tensor shape for saving: {decoded.shape}")

    # Un-normalize: Convert back to 16-bit PCM format
    decoded = (decoded * 32767).clamp(-32768, 32767).short()

    # Save the decoded audio
    target_sr = 24000  # EnCodec's default
    torchaudio.save(output_file, decoded, sample_rate=target_sr)
    print(f"Decoded audio saved to {output_file}.")





def process_audio(input_file, output_dir, model, device, b=6):
    print(f"Processing {input_file}...")

    # Load the audio
    wav, sr = torchaudio.load(input_file)

    # Resample to 24 kHz if necessary
    target_sr = 24000
    if sr != target_sr:
        print(f"Resampling from {sr} Hz to {target_sr} Hz...")
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)

    # Force mono conversion
    if wav.shape[0] > 1:
        print("Converting to mono...")
        wav = wav.mean(dim=0, keepdim=True)

    # Normalize the audio
    wav = normalize_audio(wav)

    # Move to the appropriate device
    wav = wav.unsqueeze(0).to(device)

    # Encode the audio
    print("Encoding...")
    with torch.no_grad():
        encoded = model.encode(wav, bandwidth=b)

    # Create a default audio_scales tensor (filled with ones)
    audio_codes = encoded["audio_codes"]
    audio_scales = torch.ones(audio_codes.size(0), audio_codes.size(1), device=device)

    # Decode the audio
    print("Decoding...")
    decoded = model.decode(audio_codes, audio_scales)["audio_values"].squeeze(0).cpu()
    print(f"Shape of decoded tensor: {decoded.shape}")
    print(f"Data type of decoded tensor: {decoded.dtype}")

    # Un-normalize: Convert back to 16-bit PCM format
    decoded = (decoded * 32767).clamp(-32768, 32767).short()
    print(f"Shape of decoded tensor: {decoded.shape}")
    print(f"Data type of decoded tensor: {decoded.dtype}")

    # Save the decoded file
    output_file = f"{output_dir}/decoded_mono_{b}.wav"
    torchaudio.save(output_file, decoded, sample_rate=target_sr)
    print(f"Decoded audio saved to {output_file}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode and decode audio using Facebook's EnCodec with Hugging Face transformers.")
    parser.add_argument("input", type=str, help="Path to the input WAV file")
    parser.add_argument("output", type=str, help="Path to the output file or directory")
    parser.add_argument("--encode-to-discrete", action="store_true", help="Encode the WAV file to a discrete representation and save to a text file")
    parser.add_argument("--decode-from-discrete", action="store_true", help="Decode a discrete representation from a text file to an audio file")
    
    args = parser.parse_args()
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load EnCodec model
    model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
    
    bandwidths = [1.5, 3.0, 6.0, 12.0, 24.0]
    if args.encode_to_discrete:
        # Encode to discrete representation
        for b in bandwidths:
            encode_to_discrete(args.input, args.output, model, device, b)
    elif args.decode_from_discrete:
        # Decode from discrete representation
        decode_from_discrete(args.input, args.output, model, device)
    else:
        # Perform full encode-decode process
        bandwidths = [1.5, 3.0, 6.0, 12.0, 24.0]
        for b in bandwidths:
            process_audio(args.input, args.output, model, device, b)  # Correct number of arguments
