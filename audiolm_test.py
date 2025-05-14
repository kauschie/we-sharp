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

sem_step = 51525
coarse_step = 25070
fine_step = 9000

sem_path = f"./results/semantic.transformer.{sem_step}.terminated_session.pt"
coarse_path = f"./results/coarse.transformer.{coarse_step}.terminated_session.pt"
fine_path = f"./results/fine.transformer.{fine_step}.pt"


# sem_step = 25000
# coarse_step = 29219
# fine_step = 24245


# sem_path = "./great/p1_results/semantic.transformer.25000.pt"
# coarse_path = "./great/p1_results/coarse.transformer.29219.terminated_session.pt"
# fine_path = "./great/p1_results/fine.transformer.24245.terminated_session.pt"

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
    depth=6,  # Number of transformer layers
    heads=8,
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
    heads = 8,
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
    heads = 8,
    # flash_attn = True,
).cuda()
fine_transformer.load(fine_path)

uc = False
audiolm = AudioLM(
    wav2vec = wav2vec,
    codec = encodec,
    semantic_transformer = semantic_transformer,
    coarse_transformer = coarse_transformer,
    fine_transformer = fine_transformer,
    unique_consecutive=uc,
)

import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def main():
    output_file = f"./p2_sem-{sem_step}-coarse-{coarse_step}-fine-{fine_step}-uc_{uc}"
    # output_file = "./p2_TEST"
    sample_rate = 24000  # Example: 24kHz
    
    # Generate audio using AudioLM
    # output = audiolm(batch_size=1, max_length=sample_rate*4)
    prime_path = "./seed_files/copy_1_0.5s.wav"
    # output = audiolm(batch_size=1, max_length=250, prime_wave_path=prime_path)
    output = audiolm(batch_size=6, max_length=50*6)
    # output = audiolm(batch_size=6, desired_duration=10)

    # # # Check the shape of the generated wave before processing
    # print(f"Shape of generated_wave before concatenation: {len(output) if isinstance(output, list) else output.shape}")

    print(f"type returned: {type(output)}")
    if isinstance(output, list):
        print(f"list length: {len(output)}")
        for i in range(len(output)):
            print(f"output type: {type(output[i])}")
            print(f"output[i] shape: {output[i].shape}")  # Debugging line

            # Ensure correct shape (add channel dimension if necessary)
            audio_tensor = output[i].cpu()

            if audio_tensor.dim() == 1:  # If it's (samples,), add a channel dimension
                audio_tensor = audio_tensor.unsqueeze(0)  # Convert to (1, samples)

            print(f"Processed tensor shape for saving: {audio_tensor.shape}")

            # Save the processed audio
            torchaudio.save(f"{output_file}-{i}.wav", audio_tensor, sample_rate)
            print(f"Audio successfully saved to {output_file}-{i}.wav")

    else:
        torchaudio.save(f"{output_file}.wav", output.cpu(), sample_rate)
        print(f"Audio successfully saved to {output_file}.wav")
    # torchaudio.save(output_file, normalized_wave, sample_rate)


    # Assuming you have instantiated the models:
    # print(f"Semantic Transformer Parameters: {count_parameters(semantic_transformer):,}")
    # print(f"Coarse Transformer Parameters: {count_parameters(coarse_transformer):,}")
    # print(f"Fine Transformer Parameters: {count_parameters(fine_transformer):,}")

    # # Total count
    # total_params = (count_parameters(semantic_transformer) + 
    #                 count_parameters(coarse_transformer) + 
    #                 count_parameters(fine_transformer))
    # print(f"Total Trainable Parameters in AudioLM: {total_params:,}")


main()