import subprocess
import os
os.chdir(os.path.dirname(__file__))
subprocess.run(["python", "LowMem_Ckpt_Bar_IM_QAM_4.py"])
subprocess.run(["python", "LowMem_Ckpt_Bar_IM_QAM_16.py"])
subprocess.run(["python", "LowMem_Ckpt_Bar_IM_QAM_64.py"])
