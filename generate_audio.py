from audiolm_pytorch import AudioLM
from audiolm_pytorch import SemanticTransformer
from audiolm_pytorch import CoarseTransformer
from audiolm_pytorch import FineTransformer
from audiolm_pytorch import HubertWithKmeans 
from audiolm_pytorch import EncodecWrapper
import torch
import torchaudio
import argparse
import math

hubert_checkpoint_path = "./models/hubert_base_ls960.pt"
hubert_kmeans_path = "./models/hubert_base_ls960_L9_km500.bin"

sem_path = "./results/semantic.transformer.25000.pt"
coarse_path = "./results/coarse.transformer.29219.terminated_session.pt"
fine_path = "./results/fine.transformer.24245.terminated_session.pt"

# Define and initialize the Neural Audio Codec
encodec = EncodecWrapper()

# Initialize HubertWithKmeans
wav2vec = HubertWithKmeans(
    checkpoint_path=hubert_checkpoint_path,
    kmeans_path=hubert_kmeans_path
).cuda()

# Define and initialize the Semantic Transformer
semantic_transformer = SemanticTransformer(
    num_semantic_tokens=wav2vec.codebook_size,  # From HubertWithKmeans
    dim=1024,  # Transformer dimensionality
    depth=12,  # Number of transformer layers
    heads=16,
).cuda()
semantic_transformer.load(sem_path)

# Define and initialize the Coarse Transformer
coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 1024,
    depth = 6,
    heads = 16,
).cuda()
coarse_transformer.load(coarse_path)

# Define and initialize the Fine Transformer
fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 1024,
    depth = 6,
    heads = 16,
).cuda()
fine_transformer.load(fine_path)

audiolm = AudioLM(
    wav2vec = wav2vec,
    codec = encodec,
    semantic_transformer = semantic_transformer,
    coarse_transformer = coarse_transformer,
    fine_transformer = fine_transformer,
    unique_consecutive=False
)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate stitched AudioLM output of a given duration.")
    parser.add_argument("--duration", type=float, required=True, help="Desired output duration in seconds")
    parser.add_argument("--output", type=str, default="stitched_audio", help="Output filename (without .wav)")
    parser.add_argument("--prime_wave", type=str, default=None, help="Path to WAV file to use as an initial seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save individual segments")

    args = parser.parse_args()
    desired_clip_duration = args.duration  # User-defined output length in seconds
    output_file = args.output
    prime_wave_path = args.prime_wave
    debug_mode = args.debug

    # Constants
    sample_rate = 24000
    output_length = sample_rate * 3  # 3-second chunks per generation
    overlap_length = int(sample_rate * 0.3)  # 0.3s overlap
    fade_out_duration = int(sample_rate * 0.5)  # 0.5s fade out at the end

    generated_audio = []
    debug_counter = 1  # Counter for debug file names

    # Compute fade-in and fade-out values **outside the loop** (optimization)
    fade_in = torch.linspace(0, 1, overlap_length).unsqueeze(0)
    fade_out = torch.linspace(1, 0, overlap_length).unsqueeze(0)

    # Cosine fade-out function: (1 + cos(πt)) / 2
    t = torch.linspace(0, 1, fade_out_duration)
    cosine_fade_out = (1 + torch.cos(math.pi * t)) / 2
    final_fade_out = cosine_fade_out.unsqueeze(0)

    # Load prime_wave if specified
    prime_wave = None
    if prime_wave_path:
        print(f"Loading prime wave from {prime_wave_path}...")
        prime_wave, prime_sample_rate = torchaudio.load(prime_wave_path)

        # Resample if necessary
        if prime_sample_rate != sample_rate:
            print(f"Resampling prime wave from {prime_sample_rate}Hz to {sample_rate}Hz...")
            resampler = torchaudio.transforms.Resample(orig_freq=prime_sample_rate, new_freq=sample_rate)
            prime_wave = resampler(prime_wave)

        # Ensure correct shape and move to GPU
        if prime_wave.dim() == 1:
            prime_wave = prime_wave.unsqueeze(0)
        prime_wave = prime_wave.cuda()

    # Generate first clip
    output = audiolm(batch_size=1, max_length=output_length, prime_wave=prime_wave, prime_wave_input_sample_hz=sample_rate)

    # Ensure output is a tensor (convert list if necessary)
    if debug_mode:
        print(f"type returned: {type(output)}")
    if isinstance(output, list):
        if debug_mode:
            print(f"length: {len(output)}")
        output = output[0]
        if debug_mode:
            print(f"len output vector: {len(output)}")

    if output.dim() == 1:
        output = output.unsqueeze(0)  # Ensure correct shape
    if debug_mode:
        print(f"output after unsqueeze: {output.shape}")

    # Move to CPU for saving later
    output = output.cpu()

    # Store the first part of the audio (excluding the overlap)
    generated_audio.append(output[:, :-overlap_length])

    if debug_mode:
        torchaudio.save(f"piece{debug_counter}.wav", output[:, :-overlap_length], sample_rate)
        debug_counter += 1

    # Continue generating until we reach the desired duration
    while sum([chunk.shape[1] for chunk in generated_audio]) < (desired_clip_duration * sample_rate):
        seed = output[:, -overlap_length:]  # Take the overlap as seed

        # Generate the next clip
        next_output = audiolm(batch_size=1, max_length=output_length, prime_wave=seed, prime_wave_input_sample_hz=sample_rate)

        # Ensure correct format
        if isinstance(next_output, list):
            next_output = next_output[0]
        if next_output.dim() == 1:
            next_output = next_output.unsqueeze(0)

        # Move to CPU before processing
        next_output = next_output.cpu()

        # Smooth the overlap region
        transition_start = output[:, -overlap_length:] * fade_out + next_output[:, :overlap_length] * fade_in

        # Append transition and the non-overlapping portion of next_output
        generated_audio.append(transition_start)
        generated_audio.append(next_output[:, overlap_length:])

        if debug_mode:
            torchaudio.save(f"piece{debug_counter}.wav", transition_start, sample_rate)
            debug_counter += 1
            torchaudio.save(f"piece{debug_counter}.wav", next_output[:, overlap_length:], sample_rate)
            debug_counter += 1

        # Update output for next iteration
        output = next_output


    # Concatenate all audio clips **before applying fade-out**
    final_audio = torch.cat(generated_audio, dim=-1)

    # Apply **cosine fade-out** to the last `fade_out_duration` samples
    fade_section_start = max(final_audio.shape[1] - fade_out_duration, 0)  # Ensure we don’t go negative
    final_audio[:, fade_section_start:] *= final_fade_out[:, -final_audio.shape[1] + fade_section_start:]

    # Save the final stitched audio file
    torchaudio.save(f"{output_file}.wav", final_audio, sample_rate)
    print(f"Saved final stitched audio to {output_file}.wav with duration {desired_clip_duration} seconds.")


if __name__ == "__main__":
    main()

