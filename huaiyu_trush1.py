import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import soundfile as sf
from scipy.spatial.distance import cdist

# 加载音频文件
audio_path = 'goodpiano.wav'  # 替换为你的音频文件路径
y, sr = librosa.load(audio_path, sr=None)  # 保留原音频的采样率
print(f"Sampling Rate: {sr}")

n_fft = 1024
hop_length = 1024
# 计算Mel-Spectrogram
n_mels = 256  # Mel频带的数量
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

# 输出Mel-Spectrogram的形状
print(f'Mel-Spectrogram shape: {S.shape}')  # (n_mels, time_frames)

# 转置矩阵，使每列（时间帧）作为特征向量
S_transposed = S.T  # (time_frames, n_mels)

# 聚类的数量，选择合理的聚类数
n_clusters = 150  # 可以调整为1000以保留更多细节

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(S_transposed)

# 获取每个时间帧的聚类标签
cluster_labels = kmeans.labels_

# 获取每个聚类的质心
cluster_centers = kmeans.cluster_centers_  # 形状为 (n_clusters, n_mels)

# 创建新矩阵，选择距离质心最近的实际向量作为替换
new_matrix = np.zeros_like(S_transposed)

# 遍历每个聚类
for cluster in range(n_clusters):
    # 找到当前聚类的所有时间帧索引
    cluster_indices = np.where(cluster_labels == cluster)[0]

    # 获取当前聚类中所有向量
    cluster_vectors = S_transposed[cluster_indices]

    # 计算这些向量与质心的距离
    distances = cdist(cluster_vectors, [cluster_centers[cluster]], metric='euclidean')

    # 找到距离质心最近的向量索引
    closest_index = cluster_indices[np.argmin(distances)]

    # 用该实际向量替换该类中的所有向量
    new_matrix[cluster_indices] = S_transposed[closest_index]

# 转置回原来的形状 (n_mels, time_frames)
new_matrix_transposed = new_matrix.T

# 输出新的矩阵形状
print(f'New matrix shape: {new_matrix_transposed.shape}')

# 可视化原始Mel-Spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel', fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Original Mel-Spectrogram')
plt.tight_layout()
plt.show()

# 可视化新矩阵（线性Mel-Spectrogram）
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(new_matrix_transposed, ref=np.max), sr=sr, x_axis='time', y_axis='mel',
                         fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Clustered Mel-Spectrogram (Using Closest Actual Vector as Centroid)')
plt.tight_layout()
plt.show()

# 绘制频数直方图，反映每个类中的样本数量
plt.figure(figsize=(8, 6))
plt.hist(cluster_labels, bins=n_clusters, edgecolor='black')
plt.title('Histogram of Cluster Sizes')
plt.xlabel('Cluster')
plt.ylabel('Number of Frames')
plt.grid(True)
plt.show()

# 从线性Mel频谱恢复音频信号
audio_signal = librosa.feature.inverse.mel_to_audio(new_matrix_transposed, sr=sr, n_iter=256, hop_length=hop_length)

# 将时域信号保存为WAV文件
wav_output_path = 'badpiano.wav'
sf.write(wav_output_path, audio_signal, sr)

print("done!")
