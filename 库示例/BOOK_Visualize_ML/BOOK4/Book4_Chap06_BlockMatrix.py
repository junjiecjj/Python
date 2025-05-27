


# Bk4_Ch4_05.py
import numpy as np
# define matrix
A = np.matrix([[1, 2],
                [3, 4],
                [5, 6]])
# scaler
k = 2;
# column vector c
c = np.array([[3],
                [2],
                [1]])
# row vector r
r = np.array([[2,1]])
# broadcasting principles
# matrix A plus scalar k
A_plus_k = A + k
# matrix A plus column vector c
A_plus_a = A + c
# matrix A plus row vector r
A_plus_r = A + r
# column vector c plus row vector r
c_plus_r = c + r





# Bk4_Ch4_12.py

import numpy as np

A = np.matrix([[1, 2],
               [3, 4]])

print(A.I)

B = np.array([[1, 2],
              [3, 4]])

print(B.I)



# Bk4_Ch4_13.py

import numpy as np
A = np.array([[1, -1, 0],
              [3,  2, 4],
              [-2, 0, 3]])

# calculate trace of A
tr_A = np.trace(A)








# Bk4_Ch4_14.py

import numpy as np

A = np.array([[1,2],
              [3,4]])

B = np.array([[5,6],
              [7,8]])

# Hadamard product
A_times_B_piecewise = np.multiply(A,B)
A_times_B_piecewise_V2 = A*B



# Bk4_Ch4_15.py

import numpy as np
A = np.array([[4, 2],
              [1, 3]])

# calculate determinant of A
det_A = np.linalg.det(A)




# Bk4_Ch5_01.py
import numpy as np

a = np.array([[1],
              [2],
              [3]])

a_1D = np.array([1,2,3])

b = np.array([[-4],
              [-5],
              [-6]])

b_1D = np.array([-4,-5,-6])

# sum of the elements in a

print(np.einsum('ij->',a))
print(np.einsum('i->',a_1D))

# element-wise multiplication of a and b

print(np.einsum('ij,ij->ij',a,b))
print(np.einsum('i,i->i',a_1D,b_1D))

# inner product of a and b

print(np.einsum('ij,ij->',a,b))
print(np.einsum('i,i->',a_1D,b_1D))

# outer product of a and itself

print(np.einsum('ij,ji->ij',a,a))
print(np.einsum('i,j->ij',a_1D,a_1D))


#  outer product of a and b

print(np.einsum('ij,ji->ij',a,b))
print(np.einsum('i,j->ij',a_1D,b_1D))

#

# A is a square matrix
A = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

B = np.array([[-1,-4,-7],
              [-2,-5,-8],
              [-3,-6,-9]])

# transpose of A

print(np.einsum('ji',A))

# sum of all values in A

print(np.einsum('ij->',A))

# sum across rows

print(np.einsum('ij->j',A))

# sum across columns

print(np.einsum('ij->i',A))

# extract main diagonal of A

print(np.einsum('ii->i',A))

#%% calculate the trace of A

print(np.einsum('ii->',A))

# matrix multiplication of A and B

print(np.einsum('ij,jk->ik', A, B))

# sum of all elements in the matrix multiplication of A and B

print(np.einsum('ij,jk->', A, B))

#  first matrix multiplication, then transpose

print(np.einsum('ij,jk->ki', A, B))

#  element-wise multiplication of A and B

print(np.einsum('ij,ij->ij', A, B))





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
A = np.random.random_integers(0,40,size=(6,4))
A = A/20 - 1

B = np.random.random_integers(0,40,size=(4,3))
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


# Bk4_Ch6_03.py

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





























































































































































































































































