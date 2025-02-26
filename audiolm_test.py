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

sem_path = "./results/semantic.transformer.25000.pt"
# coarse_path = "./results/coarse.transformer.200.final.pt"
# fine_path = "./results/fine.transformer.400.final.pt"

# sem_path = "./results/semantic.transformer.53.terminated_session.pt"
coarse_path = "./results/coarse.transformer.29219.terminated_session.pt"
fine_path = "./results/fine.transformer.24245.terminated_session.pt"

# sem_path = "./great/results_2s/semantic.transformer.200.pt"
# coarse_path = "./great/results_2s/coarse.transformer.200.pt"
# fine_path = "./great/results_2s/fine.transformer.400.pt"


# Define and initialize the Neural Audio Codec
encodec = EncodecWrapper()

# Initialize HubertWithKmeans
wav2vec = HubertWithKmeans(
    checkpoint_path=hubert_checkpoint_path,
    kmeans_path=hubert_kmeans_path
).cuda()

# wav2vec = HubertWithKmeans(
#     # checkpoint_path=hubert_checkpoint_path,
#     checkpoint_path=None,
#     kmeans_path=hubert_kmeans_path,
#     # use_mert=False
# ).cuda()

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
    dim = 1024,
    depth = 6,
    heads = 16,
    # flash_attn = True,
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

# print("1 Training Session / Good Quality:")

def main():
    output_file = "sem-25000-coarse-29219-fine-24245-uc_f"
    sample_rate = 24000  # Example: 24kHz
    
    # Generate audio using AudioLM
    output = audiolm(batch_size=1, max_length=sample_rate*3)

    # # # Check the shape of the generated wave before processing
    print(f"Shape of generated_wave before concatenation: {len(output) if isinstance(output, list) else output.shape}")

    print(f"type returned: {type(output)}")
    if type(output) == list:
        print(f"length: {len(output)}")
        output = output[0]
        print(f"len output vector: {len(output)}")
        if output.dim() == 1:
            output = output.unsqueeze(0)
        print(f"output after unsqueeze: {type(output)}")
        torchaudio.save(f"{output_file}.wav", output.cpu(), sample_rate)
    else:
        torchaudio.save(f"{output_file}.wav", output.cpu(), sample_rate)
    # torchaudio.save(output_file, normalized_wave, sample_rate)

    print(f"Audio successfully saved to {output_file}.wav")





    ########## or with priming

    # prime_path = "output_0-3s.wav"
    # prime_wav = audiolm(prime_wave_path=prime_path, max_length=sample_rate*4)

    # print(f"Shape of generated_wave before concatenation: {len(prime_wav) if isinstance(prime_wav, list) else prime_wav.shape}")



    # print(f"type returned: {type(prime_wav)}")
    # if type(prime_wav) == list:
    #     print(f"length: {len(prime_wav)}")
    #     prime_wav = prime_wav[0]
    #     if prime_wav.dim() == 1:
    #         prime_wav = prime_wav.unsqueeze(0)
    #     print(f"prime_wav after unsqueeze: {type(prime_wav)}")
    #     torchaudio.save(f"{output_file}-primed.wav", prime_wav.cpu(), sample_rate)
    # else:
    #     torchaudio.save(f"{output_file}-primed.wav", prime_wav.cpu(), sample_rate)

    # print(f"Audio successfully saved to {output_file}-primed.wav")

main()