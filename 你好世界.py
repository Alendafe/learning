import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

# 使用原始字符串来避免转义字符问题
file_path = r"D:\桌面\enhanced_audio.wav"

# 使用 librosa 加载音频文件
waveform, sample_rate = librosa.load(file_path, sr=16000)

# 归一化音频
normalized_waveform = librosa.util.normalize(waveform)

# 梅尔频谱特征提取
mel_spectrogram = librosa.feature.melspectrogram(y=normalized_waveform, sr=sample_rate, n_mels=128)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# 梅尔频率倒谱系数提取
mfccs = librosa.feature.mfcc(y=normalized_waveform, sr=sample_rate, n_mfcc=40)

# 将 numpy 数组转换为 PyTorch 张量
mel_spectrogram_tensor = torch.tensor(mel_spectrogram_db, dtype=torch.float32)
mfccs_tensor = torch.tensor(mfccs, dtype=torch.float32)

# 绘制波形图
plt.figure(figsize=(12, 4))
librosa.display.waveshow(normalized_waveform, sr=sample_rate)
plt.title('Normalized Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# 绘制梅尔频谱图
plt.figure(figsize=(12, 4))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Mel Bin')
plt.show()

# 绘制 MFCCs 图
plt.figure(figsize=(12, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=sample_rate, cmap='viridis')
plt.colorbar()
plt.title('MFCCs')
plt.xlabel('Time (s)')
plt.ylabel('MFCC Coefficient')
plt.show()

# 打印张量形状
print("Mel-Spectrogram Tensor Shape:", mel_spectrogram_tensor.shape)
print("MFCCs Tensor Shape:", mfccs_tensor.shape)