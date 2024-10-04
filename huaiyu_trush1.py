import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import soundfile as sf
from scipy.spatial.distance import cdist

# Load the audio file
audio_path = 'goodpiano.wav'  # Replace with your audio file path
y, sr = librosa.load(audio_path, sr=None)  # Retain the original audio sample rate
print(f"Sampling Rate: {sr}")

n_fft = 1024
hop_length = 1024
# Compute the Mel-Spectrogram
n_mels = 256  # Number of Mel bands
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

# Output the shape of the Mel-Spectrogram
print(f'Mel-Spectrogram shape: {S.shape}')  # (n_mels, time_frames)

# Transpose the matrix so that each column (time frame) becomes a feature vector
S_transposed = S.T  # (time_frames, n_mels)

# Number of clusters, choose a reasonable number of clusters
n_clusters = 150  # Can adjust to 1000 to retain more detail

# Perform clustering using KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(S_transposed)

# Get the cluster label for each time frame
cluster_labels = kmeans.labels_

# Get the centroids of each cluster
cluster_centers = kmeans.cluster_centers_  # Shape is (n_clusters, n_mels)

# Create a new matrix, selecting the actual vector closest to the centroid as a replacement
new_matrix = np.zeros_like(S_transposed)

# Loop through each cluster
for cluster in range(n_clusters):
    # Find the indices of all time frames in the current cluster
    cluster_indices = np.where(cluster_labels == cluster)[0]

    # Get all vectors in the current cluster
    cluster_vectors = S_transposed[cluster_indices]

    # Calculate the distance of these vectors from the centroid
    distances = cdist(cluster_vectors, [cluster_centers[cluster]], metric='euclidean')

    # Find the index of the vector closest to the centroid
    closest_index = cluster_indices[np.argmin(distances)]

    # Replace all vectors in this cluster with the closest actual vector
    new_matrix[cluster_indices] = S_transposed[closest_index]

# Transpose back to the original shape (n_mels, time_frames)
new_matrix_transposed = new_matrix.T

# Output the shape of the new matrix
print(f'New matrix shape: {new_matrix_transposed.shape}')

# Visualize the original Mel-Spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel', fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Original Mel-Spectrogram')
plt.tight_layout()
plt.show()

# Visualize the new matrix (Clustered Mel-Spectrogram)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(new_matrix_transposed, ref=np.max), sr=sr, x_axis='time', y_axis='mel',
                         fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Clustered Mel-Spectrogram (Using Closest Actual Vector as Centroid)')
plt.tight_layout()
plt.show()

# Plot the histogram of cluster sizes, showing the number of samples in each cluster
plt.figure(figsize=(8, 6))
plt.hist(cluster_labels, bins=n_clusters, edgecolor='black')
plt.title('Histogram of Cluster Sizes')
plt.xlabel('Cluster')
plt.ylabel('Number of Frames')
plt.grid(True)
plt.show()

# Recover the audio signal from the linear Mel-Spectrogram
audio_signal = librosa.feature.inverse.mel_to_audio(new_matrix_transposed, sr=sr, n_iter=256, hop_length=hop_length)

# Save the time-domain signal as a WAV file
wav_output_path = 'badpiano.wav'
sf.write(wav_output_path, audio_signal, sr)

print("done!")
