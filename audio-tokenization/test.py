import librosa
import numpy as np
from sklearn.cluster import KMeans
import soundfile as sf
import librosa.util
import matplotlib.pyplot as plt

def save_mel_spectrogram(y, sr, title, filename):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 1. Load the audio file
audio_path = './assets/happy-piano-melody-bright_135bpm_D_major.wav'
print('Loading audio file: ' + audio_path[9:] + '\n')
y, sr = librosa.load(audio_path)
print('********************************************')
print('Audio loaded successfully!')
print("Time Series (y): " + str(y) + "\n")
print("Sampling Rate (sr): " + str(sr) + "\n")
print('********************************************')

# Save mel spectrogram of original audio
save_mel_spectrogram(y, sr, 'Original Mel Spectrogram', './assets/original_mel_spectrogram.png')

# 2. Create an STFT spectrogram
n_fft = 2048 # n_fft is the number of samples per frame
hop_length = 512 # hop_length is the number of samples between frames
stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length) # Short-time Fourier Transform
stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max) # Convert amplitude to dB
print('Creating STFT spectrogram: ' + str(stft_db.shape) + '\n') 

# 3. Cluster the STFT spectrogram
n_clusters = 20 
kmeans = KMeans(n_clusters=n_clusters)

# Reshape stft_db for clustering
stft_db_reshaped = stft_db.reshape(-1, 1) 

# Fit and predict
clustered_stft = kmeans.fit_predict(stft_db_reshaped) 

# Reshape clustered_stft back to the original shape
clustered_stft = clustered_stft.reshape(stft_db.shape)

# Convert clustered data back to STFT scale
clustered_stft_db = np.zeros_like(stft_db)
for i in range(n_clusters):
    mask = (clustered_stft == i)
    clustered_stft_db[mask] = np.mean(stft_db[mask])

# 4. Convert the clustered STFT spectrogram back to audio
stft_clustered = librosa.db_to_amplitude(clustered_stft_db)
y_reconstructed = librosa.griffinlim(stft_clustered, n_iter=100, hop_length=hop_length)

# Scale the reconstructed audio back to the original amplitude
scale_factor = np.max(np.abs(y)) / np.max(np.abs(y_reconstructed))
y_reconstructed_normalized = y_reconstructed * scale_factor

# Save the reconstructed audio
print('********************************************')
print('STFT Reconstruction: ' + str(y_reconstructed_normalized) + '\n')
sf.write('reconstructed_audio_stft.wav', y_reconstructed_normalized, sr)

# Save mel spectrogram of reconstructed audio
save_mel_spectrogram(y_reconstructed_normalized, sr, 'Reconstructed Mel Spectrogram', './assets/reconstructed_mel_spectrogram.png')

print('********************************************')
print('Testing solution completed.')
print('Mel spectrograms have been saved in the assets folder.')