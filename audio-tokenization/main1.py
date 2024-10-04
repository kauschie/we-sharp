import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import soundfile as sf
from scipy.spatial.distance import cdist

# Load the audio file
audio_path = './assets/happy-piano-melody-bright_135bpm_D_major.wav'  # Replace with your audio file path
y, sr = librosa.load(audio_path, sr=None)  # Retain the original audio sample rate
print(f"Sampling Rate: {sr}")

n_fft = 2048
hop_length = 512
# Compute the Mel-Spectrogram
n_mels = 256  # Number of Mel bands
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

# Output the shape of the Mel-Spectrogram
print(f'Mel-Spectrogram shape: {S.shape}')  # (n_mels, time_frames)

# Transpose the matrix so that each column (time frame) becomes a feature vector
S_transposed = S.T  # (time_frames, n_mels)

# Number of clusters
n_clusters = 150

# Perform clustering using KMeans
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(S_transposed)

# Get the cluster label for each time frame
cluster_labels = kmeans.labels_

# Get the centroids of each cluster
cluster_centers = kmeans.cluster_centers_

# Create a new matrix, selecting the actual vector closest to the centroid as a replacement
new_matrix = np.zeros_like(S_transposed)

# Loop through each cluster
for cluster in range(n_clusters):
    cluster_indices = np.where(cluster_labels == cluster)[0]
    cluster_vectors = S_transposed[cluster_indices]
    distances = cdist(cluster_vectors, [cluster_centers[cluster]], metric='euclidean')
    closest_index = cluster_indices[np.argmin(distances)]
    new_matrix[cluster_indices] = S_transposed[closest_index]

# Transpose back to the original shape
new_matrix_transposed = new_matrix.T

# Save the original Mel-Spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel', fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Original Mel-Spectrogram')
plt.tight_layout()
plt.savefig('original_spectrogram.png')
plt.close()

# Save the new matrix (Clustered Mel-Spectrogram)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(new_matrix_transposed, ref=np.max), sr=sr, x_axis='time', y_axis='mel',
                         fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Clustered Mel-Spectrogram (Using Closest Actual Vector as Centroid)')
plt.tight_layout()
plt.savefig('clustered_spectrogram.png')
plt.close()

# Save the histogram of cluster sizes
plt.figure(figsize=(8, 6))
plt.hist(cluster_labels, bins=n_clusters, edgecolor='black')
plt.title('Histogram of Cluster Sizes')
plt.xlabel('Cluster')
plt.ylabel('Number of Frames')
plt.grid(True)
plt.savefig('cluster_histogram.png')
plt.close()

# Recover and save the audio signal
audio_signal = librosa.feature.inverse.mel_to_audio(new_matrix_transposed, sr=sr, n_iter=256, hop_length=hop_length)
wav_output_path = 'badpiano.wav'
sf.write(wav_output_path, audio_signal, sr)

print("All files have been saved!")