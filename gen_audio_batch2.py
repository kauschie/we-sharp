#
#   Example usage statements:

# python gen_audio_batch2.py --duration 8 --batch_size 4 --prime_wave seed_files/midi_cut.wav --output my_generated_track

#           will generate an 4 x ~8 second clips and change the output 
#               name to my_generated_track.wav
#
#

# python gen_audio_batch2.py --duration 8 --batch_size 4 --output my_generated_track

#           will generate an 4 x ~8 second clips and change the output 
#               name to my_generated_track.wav
#
#


# python gen_audio_batch2.py --duration 10

#           will generate at least 10 seconds of audio, likely a bit more 
#               because i don't slice it at 10 exactly i just make sure that 
#               it finishes after the most recent one that gets it past 10 seconds.

# python gen_audio_batch2.py --duration 10 --prime_wave seed.wav --output my_track

#       generates ~10s audio with see input seed.wav which is the path 
#               to the wav input pile used as a seed

# python gen_audio_batch2.py --duration 10 --debug

#           will enable debug mode which prints some stuff out to look at 
#               tensor shapes at different points and outputs the slices
#                   of music generated

# python gen_audio_batch2.py --duration 8 --output my_generated_track

#           will generate an ~8 second clip and change the output 
#               name to my_generated_track.wav
#

from audiolm_pytorch import AudioLM, SemanticTransformer, CoarseTransformer, FineTransformer, HubertWithKmeans, EncodecWrapper
import torch
import torchaudio
import argparse
import math
from pathlib import Path
import os

# Model paths
hubert_checkpoint_path = "./models/hubert_base_ls960.pt"
hubert_kmeans_path = "./models/hubert_base_ls960_L9_km500.bin"

sem_step = 55000
coarse_step = 108507
fine_step = 136325

# sem_path = f"./results/semantic.transformer.{sem_step}.pt"
# coarse_path = f"./results/coarse.transformer.{coarse_step}.terminated_session.pt"
# fine_path = f"./results/fine.transformer.{fine_step}.terminated_session.pt"

sem_path = "./great/p1_results/semantic.transformer.25000.pt"
coarse_path = "./great/p1_results/coarse.transformer.29219.terminated_session.pt"
fine_path = "./great/p1_results/fine.transformer.24245.terminated_session.pt"



# Initialize models
encodec = EncodecWrapper()
wav2vec = HubertWithKmeans(
    checkpoint_path=hubert_checkpoint_path, 
    kmeans_path=hubert_kmeans_path
    ).cuda()

semantic_transformer = SemanticTransformer(
    num_semantic_tokens=wav2vec.codebook_size, 
    dim=1024, 
    depth=12, 
    heads=16
    ).cuda()
semantic_transformer.load(sem_path)

coarse_transformer = CoarseTransformer(
    num_semantic_tokens=wav2vec.codebook_size, 
    codebook_size=1024, 
    num_coarse_quantizers=3, 
    dim=1024, 
    depth=6, 
    heads=16
    ).cuda()
coarse_transformer.load(coarse_path)

fine_transformer = FineTransformer(
    num_coarse_quantizers=3, 
    num_fine_quantizers=5, 
    codebook_size=1024, 
    dim=1024, 
    depth=6, 
    heads=16
    ).cuda()
fine_transformer.load(fine_path)

audiolm = AudioLM(
    wav2vec=wav2vec, 
    codec=encodec, 
    semantic_transformer=semantic_transformer, 
    coarse_transformer=coarse_transformer, 
    fine_transformer=fine_transformer, 
    unique_consecutive=True
    )

# Helper to check if all tracks reached desired duration
def is_done(generated_audio, sample_rate, desired_clip_duration):
    durations = []
    for i, track in enumerate(generated_audio):
        track_len = sum(chunk.shape[1] for chunk in track)
        duration_sec = track_len / sample_rate
        durations.append(duration_sec)
        print(f"[DEBUG] Track {i} current duration: {duration_sec:.2f} seconds")
    return all(d >= desired_clip_duration for d in durations)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate stitched AudioLM output with batch processing.")
    parser.add_argument("--duration", type=int, required=True, help="Desired output duration in seconds")
    parser.add_argument("--output", type=str, default="stitched_audio", help="Output filename (without .wav)")
    parser.add_argument("--prime_wave", type=str, default=None, help="Path to WAV file to use as an initial seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save individual segments")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for parallel audio generation")


    args = parser.parse_args()
    desired_clip_duration = args.duration
    output_file = str(Path(args.output).resolve())
    prime_wave_path = args.prime_wave
    debug_mode = args.debug
    batch_size = args.batch_size

    output_dir = os.path.dirname(output_file) or os.getcwd()
    progress_path = os.path.join(output_dir, "progress.json")

    sample_rate = 24000
    # token_rate = 50
    # output_length = desired_clip_duration * token_rate
    # output_length = desired_clip_duration * token_rate
    output_length = desired_clip_duration + 1
    max_seed_seconds = 3
    max_seed_samples = int(sample_rate * max_seed_seconds)
    overlap_length = int(sample_rate * 1)
    fade_out_duration = int(sample_rate * 0.5)

    fade_in = torch.linspace(0, 1, overlap_length).unsqueeze(0)
    fade_out = torch.linspace(1, 0, overlap_length).unsqueeze(0)
    final_fade_out = ((1 + torch.cos(math.pi * torch.linspace(0, 1, fade_out_duration))) / 2).unsqueeze(0)

    generated_audio = [[] for _ in range(batch_size)]

    # Load and preprocess prime_wave
    prime_wave = None
    if prime_wave_path:
        # Load the waveform and its sample rate
        prime_wave, prime_sample_rate = torchaudio.load(prime_wave_path)

        # Resample if needed
        if prime_sample_rate != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=prime_sample_rate, new_freq=sample_rate)
            prime_wave = resampler(prime_wave)

        # Ensure the waveform is 2D (channels x samples)
        if prime_wave.dim() == 1:
            prime_wave = prime_wave.unsqueeze(0)

        # If it's stereo (2 channels), average to make it mono
        if prime_wave.size(0) > 1:
            prime_wave = prime_wave.mean(dim=0, keepdim=True)

        # Move to GPU
        prime_wave = prime_wave.cuda()


    # Generate first batch
    if prime_wave is not None:
        total_samples = prime_wave.shape[-1]
        seed_length = min(total_samples, max_seed_samples)
        seed = prime_wave[:, -seed_length:]
        prime_id = audiolm.semantic.wav2vec(seed, flatten=False, input_sample_hz=sample_rate)
        prime_ids = torch.cat([prime_id] * batch_size, dim=0)
        # output = audiolm(batch_size=batch_size, max_length=output_length, prime_ids=prime_ids, track_progress=progress_path)
        output = audiolm(batch_size=batch_size, desired_duration=output_length, prime_ids=prime_ids, track_progress=progress_path)

        for i in range(batch_size):
            generated_audio[i].append(prime_wave.cpu())
    else:
        # output = audiolm(batch_size=batch_size, max_length=output_length, track_progress=progress_path)
        output = audiolm(batch_size=batch_size, desired_duration=output_length, track_progress=progress_path)

    output = [o.cpu().unsqueeze(0) if o.dim() == 1 else o.cpu() for o in output]

    for i in range(batch_size):
        generated_audio[i].append(output[i][:, :-overlap_length])
        print(f"[DEBUG] Initial generated_audio[{i}] shape: {generated_audio[i][-1].shape}")

    # Loop to continue generation
    # Generation loop
    while not is_done(generated_audio, sample_rate, desired_clip_duration):
        print(f"\n[DEBUG] Starting new generation round")

        prime_ids = []
        for i in range(batch_size):
            last_segment = generated_audio[i][-1][:, -overlap_length:].to("cuda")

            if last_segment.shape[1] < overlap_length:
                pad_size = overlap_length - last_segment.shape[1]
                last_segment = torch.nn.functional.pad(last_segment, (pad_size, 0))

            prime_id = audiolm.semantic.wav2vec(last_segment, flatten=False, input_sample_hz=sample_rate)
            prime_ids.append(prime_id)

        prime_ids = torch.cat(prime_ids, dim=0)

        # next_output = audiolm(batch_size=batch_size, max_length=output_length, prime_ids=prime_ids, track_progress=progress_path)
        next_output = audiolm(batch_size=batch_size, desired_duration=output_length, prime_ids=prime_ids, track_progress=progress_path)
        next_output = [o.cpu().unsqueeze(0) if o.dim() == 1 else o.cpu() for o in next_output]

        min_next_output_length = min(o.shape[1] for o in next_output)
        next_output = [o[:, :min_next_output_length] for o in next_output]

        for i in range(batch_size):
            last_segment = generated_audio[i][-1][:, -overlap_length:].cuda()
            next_segment = next_output[i][:, :overlap_length].cuda()

            if last_segment.shape[1] < overlap_length:
                pad_size = overlap_length - last_segment.shape[1]
                last_segment = torch.nn.functional.pad(last_segment, (pad_size, 0))

            fade_in_adj = fade_in.cuda()
            fade_out_adj = fade_out.cuda()

            transition_start = last_segment * fade_out_adj + next_segment * fade_in_adj

            # Only append the non-overlapping portion
            generated_audio[i].append(next_output[i][:, overlap_length:].cuda())
            print(f"[DEBUG] Track {i} updated length after appending: {sum(chunk.shape[1] for chunk in generated_audio[i]) / sample_rate:.2f} seconds")

        output = next_output

    final_audio = [torch.cat([chunk.cpu() for chunk in track], dim=-1) for track in generated_audio]

    for i in range(batch_size):
        fade_start = max(final_audio[i].shape[1] - fade_out_duration, 0)
        final_audio[i][:, fade_start:] *= final_fade_out[:, -final_audio[i].shape[1] + fade_start:]

        torchaudio.save(f"{output_file}-{i}.wav", final_audio[i].cpu(), sample_rate)
        print(f"Saved final stitched audio to {output_file}-{i}.wav with duration {final_audio[i].shape[1] / sample_rate:.2f} seconds.")

if __name__ == "__main__":
    main()
