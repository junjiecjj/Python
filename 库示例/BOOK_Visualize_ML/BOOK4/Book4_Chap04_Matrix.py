

#%% Bk4_Ch4_01.py

import numpy as np

# 2d matrix
A_matrix = np.matrix([[2,4],
                      [6,8]])
print(A_matrix.shape)
print(type(A_matrix))

# 1d array
A_1d = np.array([2,4])
print(A_1d.shape)
print(type(A_1d))

# 2d array
A_2d = np.array([[2,4],
                 [6,8]])
print(A_2d.shape)
print(type(A_2d))

# 3d array
A1 = [[2,4],
      [6,8]]

A2 = [[1,3],
      [5,7]]

A3 = [[1,0],
      [0,1]]
A_3d = np.array([A1,A2,A3])
print(A_3d.shape)
print(type(A_3d))



#%% Bk4_Ch4_02.py

import numpy as np

A = np.matrix([[1,2,3],
              [4,5,6],
              [7,8,9]])

# extract diagonal elements
a = np.diag(A)

# construct a diagonal matrix
A_diag = np.diag(a)


#%% Bk4_Ch4_03.py

import numpy as np

# define matrix
A = np.matrix([[1, 2], [3, 4]])
B = np.matrix([[2, 6], [4, 8]])

# matrix addition
A_plus_B = np.add(A,B)
A_plus_B_2 = A + B


# matrix subtraction
A_minus_B = np.subtract(A,B)
A_minus_B_2 = A - B


#%% Bk4_Ch4_04.py

import numpy as np

k = 2
X = [[1,2],
     [3,4]]

# scalar multiplication
k_times_X = np.dot(k,X)
k_times_X_2 = k*np.matrix(X)


#%% Bk4_Ch4_05.py

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



#%% Bk4_Ch4_06.py

import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[2, 4],
              [1, 3]])

# matrix multiplication
A_times_B = np.matmul(A, B)
A_times_B_2 = A@B



#%% Bk4_Ch4_07.py

import numpy as np

A = np.array([[1, 2]])

B = np.array([[5, 6],
              [8, 9]])

print(A*B)

A = np.array([[1, 2]])

B = np.matrix([[5, 6],
              [8, 9]])

print(A*B)

A = np.matrix([[1, 2]])

B = np.matrix([[5, 6],
              [8, 9]])

print(A*B)



#%% Bk4_Ch4_08.py

from numpy.linalg import matrix_power as pw
A = np.array([[1., 2.],
              [3., 4.]])

# matrix inverse
A_3 = pw(A,3)
A_3_v3 = A@A@A

# piecewise power
A_3_piecewise = A**3




#%% Bk4_Ch4_09.py

import numpy as np

A = np.matrix([[1,3],
               [2,4]])

print(A**2)

B = np.array([[1,3],
              [2,4]])

print(B**2)




#%% Bk4_Ch4_10.py

import numpy as np

A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# matrix transpose
A_T = A.transpose()
A_T_2 = A.T





#%% Bk4_Ch4_11.py

from numpy.linalg import inv
A = np.array([[1., 2.],
              [3., 4.]])

# matrix inverse
A_inverse = inv(A)
A_times_A_inv = A@A_inverse






#%% Bk4_Ch4_12.py

import numpy as np

A = np.matrix([[1, 2],
               [3, 4]])

# print(A.I)

B = np.array([[1, 2],
              [3, 4]])

# print(B.I)



#%% Bk4_Ch4_13.py

import numpy as np
A = np.array([[1, -1, 0],
              [3,  2, 4],
              [-2, 0, 3]])

# calculate trace of A
tr_A = np.trace(A)



#%% Bk4_Ch4_14.py

import numpy as np

A = np.array([[1,2],
              [3,4]])

B = np.array([[5,6],
              [7,8]])

# Hadamard product
A_times_B_piecewise = np.multiply(A,B)
A_times_B_piecewise_V2 = A*B







#%% Bk4_Ch4_15.py

import numpy as np
A = np.array([[4, 2],
              [1, 3]])

# calculate determinant of A
det_A = np.linalg.det(A)































































































































































































































































