import subprocess
import os

# 获取当前脚本所在目录
base_dir = os.path.dirname(__file__)

# 路径拼接
tdm_run = os.path.join(base_dir, "TDM", "Run.py")
ddm_run = os.path.join(base_dir, "DDM", "Run.py")

# 先运行 DDM 的 Run.py
subprocess.run(["python", ddm_run])

# 再运行 TDM 的 Run.py
subprocess.run(["python", tdm_run])
