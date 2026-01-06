import torch
import numpy as np
import pandas as pd  # 如果想把结果存为 CSV
import os
os.chdir(os.path.dirname(__file__))
#0. file definition
ckpt_file = "mi_IM_64.pt"  # 你要读取的 checkpoint 文件名
save_file = "mi_IM_64.csv"  # 保存结果的文件名

# 1. 载入 checkpoint
ckpt = torch.load(ckpt_file, map_location="cpu")

# 2. 查看顶层键
print("Keys in checkpoint:", ckpt.keys())
# 比如通常会看到: dict_keys(['mi_all', 'params'])

# 3. 取出互信息矩阵和参数
mi_all = ckpt["mi_all"]      # torch.Tensor, shape (n_realizations, n_snr)
params = ckpt["params"]      # 存储 demo 时的各项参数字典

print("mi_all shape:", mi_all.shape)
print("Run parameters:")
for k, v in params.items():
    print(f"  {k} = {v}")

# 4. 转成 NumPy，或直接保存为 CSV 方便 Excel 打开
mi_np = mi_all.numpy()   # 先转为 NumPy 数组
print("mi_all as NumPy:", mi_np)

# 如果你想把每一行（每个 realization）或每列（每个 SNR）的数据保存：
df = pd.DataFrame(mi_np, columns=[f"SNR={d}dB" for d in params["snr_db"]])
df.to_csv(save_file, index_label="realization")
print(f"Saved results to {save_file}")
