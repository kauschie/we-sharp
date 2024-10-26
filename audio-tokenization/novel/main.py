import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import soundfile as s
from scipy.spatial.distance import cdist

# Load the audio file
audio_path = './assets/output_other.mp3'
y, sr = librosa.load(audio_path, sr=None)
print(f"Sampling Rate: {sr}")

# Parameters
n_fft = 2048  # Increased for better frequency resolution
hop_length = 512  # Decreased for better time resolution
window = 'hann'  # Specify window type

# Compute the STFT (complex spectrogram)
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
magnitude, phase = librosa.magphase(D)

# Convert to mel scale
n_mels = 256
mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
mel_magnitude = np.dot(mel_basis, magnitude)

# Prepare data for clustering
S_db = librosa.amplitude_to_db(mel_magnitude, ref=np.max)
S_normalized = (S_db - S_db.min()) / (S_db.max() - S_db.min())
S_transposed = S_normalized.T

# Increase number of clusters
n_clusters = 100  # Increased for better detail preservation

# Perform clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(S_transposed)
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Create a new matrix with a smoother transition between clusters
new_matrix = np.zeros_like(S_transposed)
for cluster in range(n_clusters):
    cluster_indices = np.where(cluster_labels == cluster)[0]
    cluster_vectors = S_transposed[cluster_indices]
    
    if len(cluster_vectors) > 0:
        # Calculate the mean vector for this cluster instead of using the closest vector
        mean_vector = np.mean(cluster_vectors, axis=0)
        new_matrix[cluster_indices] = mean_vector

# Convert back to original scale
new_matrix_transposed = new_matrix.T
S_db_reconstructed = (new_matrix_transposed * (S_db.max() - S_db.min())) + S_db.min()
S_reconstructed = librosa.db_to_amplitude(S_db_reconstructed)

# Inverse mel transform to get back to linear frequency scale
magnitude_reconstructed = np.dot(np.linalg.pinv(mel_basis), S_reconstructed)

# Reconstruct complex spectrogram using original phase
D_reconstructed = magnitude_reconstructed * phase

# Inverse STFT to get audio signal
audio_signal = librosa.istft(D_reconstructed, hop_length=hop_length, window=window)

# Save reconstructed audio
wav_output_path = './assets/reconstructed_piano.wav'
sf.write(wav_output_path, audio_signal, sr)

# Save spectrograms
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max), 
                         sr=sr, hop_length=hop_length, y_axis='hz', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Original Spectrogram')

plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(magnitude_reconstructed, ref=np.max), 
                         sr=sr, hop_length=hop_length, y_axis='hz', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Reconstructed Spectrogram')
plt.tight_layout()
plt.savefig('spectrogram_comparison.png')
plt.close()

print("Processing complete! Audio and spectrograms have been saved.")