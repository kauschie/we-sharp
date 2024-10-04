import librosa
import numpy as np
from sklearn.cluster import KMeans
import soundfile as sf
import librosa.util

# 1. Load the audio file
audio_path = './assets/happy-piano-melody-bright_135bpm_D_major.wav'
print('Loading audio file: ' + audio_path[9:] + '\n')
y, sr = librosa.load(audio_path)
print ('********************************************')
print ('Audio loaded successfully!')
print ("Time Series (y): " + str(y) + "\n")
print ("Sampling Rate (sr): " + str(sr) + "\n")
print ('********************************************')

# 2. Create a mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
print ('Creating mel spectrogram: ' + str(mel_spectrogram) + '\n')
mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# 3. Cluster the mel spectrogram
n_clusters = 10  # Adjust this based on your needs
kmeans = KMeans(n_clusters=n_clusters)

# Reshape mel_db for clustering
mel_db_reshaped = mel_db.reshape(-1, 1)

# Fit and predict
clustered_mel = kmeans.fit_predict(mel_db_reshaped)

# Reshape clustered_mel back to the original shape
clustered_mel = clustered_mel.reshape(mel_db.shape)

# Convert clustered data back to mel scale
clustered_mel_db = np.zeros_like(mel_db)
for i in range(n_clusters):
    mask = (clustered_mel == i)
    clustered_mel_db[mask] = np.mean(mel_db[mask])

# 4. Convert the clustered mel spectrogram back to audio
mel_clustered = librosa.db_to_power(clustered_mel_db)
y_reconstructed_inverse = librosa.feature.inverse.mel_to_audio(mel_clustered, sr=sr)
y_reconstructed_griffin_lim = librosa.griffinlim(mel_clustered)

# Scale the reconstructed audio back to the original amplitude
scale_factor = np.max(np.abs(y)) / np.max(np.abs(y_reconstructed_inverse))
y_reconstructed_inverse_normalized = y_reconstructed_inverse * scale_factor
scale_factor = np.max(np.abs(y)) / np.max(np.abs(y_reconstructed_griffin_lim))
y_reconstructed_griffin_lim_normalized = y_reconstructed_griffin_lim * scale_factor


# Save the reconstructed audio from inverse Fourier transform
print ('********************************************')
print('Inverse Fourier Reconstruction: ' + str(y_reconstructed_inverse_normalized) + '\n')
sf.write('reconstructed_audio.wav', y_reconstructed_inverse_normalized, sr)

# Save the reconstructed audio from griffin-lim algorithm
print ('********************************************')
print('Griffin-Lim Reconstruction: ' + str(y_reconstructed_griffin_lim_normalized) + '\n')
sf.write('reconstructed_audio_griffin_lim.wav', y_reconstructed_griffin_lim_normalized, sr)


print ('********************************************')
print ('Testing solution this may take a while...')
# step1 - converting a wav file to numpy array and then converting that to mel-spectrogram
my_audio_as_np_array, my_sample_rate= librosa.load("./assets/happy-piano-melody-bright_135bpm_D_major.wav")

# step2 - converting audio np array to spectrogram
spec = librosa.feature.melspectrogram(y=my_audio_as_np_array,
                                        sr=my_sample_rate, 
                                            n_fft=2048, 
                                            hop_length=512, 
                                            win_length=None, 
                                            window='hann', 
                                            center=True, 
                                            pad_mode='reflect', 
                                            power=2.0,
                                     n_mels=128)

# step3 converting mel-spectrogrma back to wav file
res = librosa.feature.inverse.mel_to_audio(spec, 
                                           sr=my_sample_rate, 
                                           n_fft=2048, 
                                           hop_length=512, 
                                           win_length=None, 
                                           window='hann', 
                                           center=True, 
                                           pad_mode='reflect', 
                                           power=2.0, 
                                           n_iter=32)

# step4 - save it as a wav file
sf.write("test1.wav", res, my_sample_rate)