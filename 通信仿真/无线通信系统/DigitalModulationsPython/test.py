import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 参数设置
duration = 2.0          # 持续时间（秒）
sample_rate = 44100     # 采样率（Hz）
frequency = 440.0       # 声音频率（Hz），标准 A4 音

# 生成时间轴
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# 生成正弦波信号（值范围 -1 到 1）
signal = 0.5 * np.sin(2 * np.pi * frequency * t)

# 可选：保存为 WAV 文件（需要将信号转换为 16-bit PCM）
wavfile.write('generated_audio.wav', sample_rate, (signal * 32767).astype(np.int16))
print("音频已保存为 generated_audio.wav")

# 绘制波形（显示前 0.05 秒以便清晰观察）
plt.figure(figsize=(12, 4))
plt.plot(t[:2205], signal[:2205])  # 2205 个点 ≈ 0.05 秒
plt.title("音频波形图 (前 0.05 秒)")
plt.xlabel("时间 [秒]")
plt.ylabel("振幅")
plt.grid(True)
plt.show()
