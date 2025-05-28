


#%% Bk4_Ch6_01.py

import numpy as np

A = np.array([[1, 2, 3, 0,  0],
              [4, 5, 6, 0,  0],
              [0, 0, 0, -1, 0],
              [0, 0 ,0, 0,  1]])

# NumPy array slicing

A_1_1 = A[0:2,0:3]

A_1_2 = A[0:2,3:]
# A_1_2 = A[0:2,-2:]
A_2_1 = A[2:,0:3]
# A_2_1 = A[-2:,0:3]
A_2_2 = A[2:,3:]
# A_2_2 = A[-2:,-2:]

# Assemble a matrix from nested lists of blocks
# 分块矩阵的意思是把若干个矩阵当成元素进行组合，从而构造出更大的矩阵，和线性代数中的分块矩阵思想相同
A_ = np.block([[A_1_1, A_1_2],
               [A_2_1, A_2_2]])


#%% Bk4_Ch6_02.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_heatmap(x,title):
    fig, ax = plt.subplots()
    ax = sns.heatmap(x,
                     cmap='RdYlBu_r',
                     cbar_kws={"orientation": "horizontal"}, vmin=-1, vmax=1)
    ax.set_aspect("equal")
    plt.title(title)

# Generate matrices A and B
A = np.random.randint(0,40,size=(6,4))
A = A/20 - 1

B = np.random.randint(0,40,size=(4,3))
B = B/20 - 1

# visualize matrix A and B
plot_heatmap(A,'A')

plot_heatmap(B,'B')

# visualize A@B
C = A@B
plot_heatmap(C,'C = AB')

C_rep = np.zeros_like(C)

# reproduce C

for i in np.arange(4):
    C_i = A[:,[i]]@B[[i],:];
    title = 'C' + str(i + 1)
    plot_heatmap(C_i,title)

    C_rep = C_rep + C_i

# Visualize reproduced C
plot_heatmap(C_rep,'C reproduced')


#%% Bk4_Ch6_03.py

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

A = np.array([[-1,  1],
              [0.7, -0.4]])

B = np.array([[0.5, -0.6],
             [-0.8, 0.3]])

A_kron_B = np.kron(A, B)

fig, axs = plt.subplots(1, 5, figsize=(12, 5))

plt.sca(axs[0])
ax = sns.heatmap(A,cmap='RdYlBu_r',vmax = 1,vmin = -1, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('A')

plt.sca(axs[1])
plt.title('$\otimes$')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(B, cmap='RdYlBu_r', vmax = 1, vmin = -1,  cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('B')

plt.sca(axs[3])
plt.title('=')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(A_kron_B, cmap='RdYlBu_r', vmax = 1, vmin = -1, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('C')





























































































































































































































































