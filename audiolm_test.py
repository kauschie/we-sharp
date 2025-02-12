from audiolm_pytorch import AudioLM
from audiolm_pytorch import SemanticTransformer
from audiolm_pytorch import CoarseTransformer
from audiolm_pytorch import FineTransformer
from audiolm_pytorch import HubertWithKmeans 
from audiolm_pytorch import EncodecWrapper
import torch
import torchaudio

hubert_checkpoint_path = "./models/hubert_base_ls960.pt"
hubert_kmeans_path = "./models/hubert_base_ls960_L9_km500.bin"
sem_path = "./results/semantic.transformer.49.final.pt"
coarse_path = "./results/coarse.transformer.64.final.pt"
fine_path = "./results/fine.transformer.77.terminated_session.pt"

# Define and initialize the Neural Audio Codec
encodec = EncodecWrapper()

# Initialize HubertWithKmeans
# wav2vec = HubertWithKmeans(
#     checkpoint_path=hubert_checkpoint_path,
#     kmeans_path=hubert_kmeans_path
# ).cuda()

wav2vec = HubertWithKmeans(
    # checkpoint_path=hubert_checkpoint_path,
    checkpoint_path=None,
    kmeans_path=hubert_kmeans_path,
    use_mert=True
).cuda()

# Define and initialize the Semantic Transformer
semantic_transformer = SemanticTransformer(
    num_semantic_tokens=wav2vec.codebook_size,  # From HubertWithKmeans
    dim=1024,  # Transformer dimensionality
    depth=12,  # Number of transformer layers
    heads=16,
    # flash_attn=True,  # Use Flash Attention for efficiency
).cuda()
semantic_transformer.load(sem_path)

# Define and initialize the Coarse Transformer
coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 512,
    depth = 6,
    # flash_attn = True,
).cuda()
coarse_transformer.load(coarse_path)

# Define and initialize the Fine Transformer
fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6,
    # flash_attn = True,
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

# Generate audio using AudioLM
generated_wave = audiolm(batch_size=1)

# Check the shape of the generated wave before processing
# print(f"Shape of generated_wave before concatenation: {len(generated_wave) if isinstance(generated_wave, list) else generated_wave.shape}")

# Concatenate the list of 1D tensors into a single 1D tensor
# if isinstance(generated_wave, list):
#     generated_wave = torch.cat(generated_wave, dim=0)

# Check the shape after concatenation
# print(f"Shape of generated_wave after concatenation: {generated_wave.shape}")

# Move the tensor to CPU for saving
# generated_wave = generated_wave.detach().cpu()

# Check the shape after moving to CPU
# print(f"Shape of generated_wave after moving to CPU: {generated_wave.shape}")

# Normalize the waveform to the range [-1, 1]
# normalized_wave = generated_wave / torch.max(torch.abs(generated_wave))
# normalized_wave = generated_wave

# Check the shape after normalization
# print(f"Shape of normalized_wave after normalization: {normalized_wave.shape}")

# Set the sample rate (adjust based on your model's configuration)
sample_rate = 24000  # Example: 24kHz

# Save the normalized waveform as a .wav file
output_file = "generated_audio.wav"
# output_file2 = "generated_audio2.wav"
torchaudio.save(output_file, generated_wave.cpu(), sample_rate)
# torchaudio.save(output_file, normalized_wave, sample_rate)

print(f"Audio successfully saved to {output_file}")

# # or with priming
# prime_path = "./p2-data/smallest_test/beach (2).wav_seg8.wav"

# generated_wav_with_prime = audiolm(prime_wave_path=prime_path)
# generated_wav_with_prime = generated_wav_with_prime.detach().cpu()
# torchaudio.save(output_file2, generated_wav_with_prime, sample_rate)

# print(f"Audio successfully saved to {output_file2}")
