
#%% Bk4_Ch2_01.py

import numpy as np
import matplotlib.pyplot as plt

def draw_vector(vector,RBG):
    array = np.array([[0, 0, vector[0], vector[1]]])
    X, Y, U, V = zip(*array)
    plt.quiver(X, Y, U, V,angles='xy', scale_units='xy',scale=1,color = RBG)

fig, ax = plt.subplots()

draw_vector([4,3],np.array([0,112,192])/255)
draw_vector([-3,4],np.array([255,0,0])/255)

plt.ylabel('$x_2$')
plt.xlabel('$x_1$')
plt.axis('scaled')
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.show()

#%% Bk4_Ch2_02.py

import numpy as np

# define two column vectors
a = np.array([[4], [3]])
b = np.array([[-3], [4]])

# calculate L2 norm
a_L2_norm = np.linalg.norm(a)
b_L2_norm = np.linalg.norm(b)



#%% Bk4_Ch1_03.py

import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(-10, 10, num=201);
x2 = x1;

xx1, xx2 = np.meshgrid(x1,x2)
p = 2
zz = ((np.abs((xx1))**p) + (np.abs((xx2))**p))**(1./p)

fig, ax = plt.subplots(figsize=(12, 12))

ax.contour(xx1, xx2, zz, levels = np.arange(11), cmap='RdYlBu_r')

ax.axhline(y=0, color='k', linewidth = 0.25)
ax.axvline(x=0, color='k', linewidth = 0.25)
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal', adjustable='box')
plt.show()

#%% Bk4_Ch1_04.py

import numpy as np

# define two column vectors
a = np.array([[-2], [5]])
b = np.array([[5], [-1]])

# calculate vector addition
a_plus_b = a + b
a_plus_b_2 = np.add(a,b)

# calculate vector subtraction
a_minus_b = a - b
a_minus_b_2 = np.subtract(a,b)

b_minus_a = b - a
b_minus_a_2 = np.subtract(b,a)




#%% Bk4_Ch2_05.py

import numpy as np

# define a column vector
a = np.array([[2], [2]])

b = 2*a
c = -1.5*a


#%% Bk4_Ch2_06.py

import numpy as np

a = np.array([[4, 3]])
b = np.array([[5, -2]])

a_dot_b = np.inner(a, b)

a_2 = np.array([[4], [3]])
b_2 = np.array([[5], [-2]])
a_dot_b_2 = a_2.T@b_2





#%% Bk4_Ch2_07.py

import numpy as np
a = np.array([[2,3],
              [3,4]])

b = np.array([[3,4],
              [5,6]])

ab = np.dot(a,b)
# a@b


#%% Bk4_Ch2_08.py

import numpy as np
a = np.array([[1,2],
              [3,4]])

b = np.array([[3,4],
              [5,6]])

a_dot_b = np.vdot(a,b)
# [1,2,3,4]*[3,4,5,6].T

#%% Bk4_Ch2_09.py

import numpy as np

a, b = np.array([[4], [3]]), np.array([[5], [-2]])

# calculate cosine theta
cos_theta = (a.T @ b) / (np.linalg.norm(a,2) * np.linalg.norm(b,2))

# calculate theta in radian
cos_radian = np.arccos(cos_theta)

# convert radian to degree
cos_degree = cos_radian * ((180)/np.pi)






#%% Bk4_Ch2_10.py

from scipy.spatial import distance
from sklearn import datasets
import numpy as np

# import the iris data
iris = datasets.load_iris()

# Only use the first two features: sepal length, sepal width
X = iris.data[:, :]

# Extract 4 data points
x1_data = X[0,:]
x2_data = X[1,:]
x51_data = X[50,:]
x101_data = X[100,:]

# calculate cosine distance
x1_x2_cos_dist = distance.cosine(x1_data,x2_data)
x1_norm = np.linalg.norm(x1_data)
x2_norm = np.linalg.norm(x2_data)
x1_dot_x2 = x1_data.T@x2_data
x1_x2_cos = x1_dot_x2/x1_norm/x2_norm


x1_x51_cos_dist = distance.cosine(x1_data,x51_data)

x1_x101_cos_dist = distance.cosine(x1_data,x101_data)





#%% Bk4_Ch2_11.py

import numpy as np
a = np.array([-2, 1, 1])
b = np.array([1, -2, -1])
# a = [-2, 1, 1]
# b = [1, -2, -1]

# calculate cross product of row vectors
a_cross_b = np.cross(a, b)
print(f"a_cross_b = {a_cross_b}")

a_col = np.array([[-2], [1], [1]])
b_col = np.array([[1], [-2], [-1]])

# calculate cross product of column vectors
a_cross_b_col = np.cross(a_col, b_col, axis=0)
print(f"a_cross_b_col = {a_cross_b_col}")


#%% Bk4_Ch2_12.py

import numpy as np
a = np.array([-2, 1, 1])
b = np.array([1, -2, -1])
# a = [-2, 1, 1]
# b = [1, -2, -1]


# calculate element-wise product of row vectors
a_times_b = np.multiply(a, b)
a_times_b_2 = a*b

a_col = np.array([[-2], [1], [1]])
b_col = np.array([[1], [-2], [-1]])

# calculate element-wise product of column vectors
a_times_b_col = np.multiply(a_col, b_col)
a_times_b_col_2 = a_col*b_col



#%% Bk4_Ch2_13.py

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

a = np.array([[0.5],[-0.7],[1],[0.25],[-0.6],[-1]])
b = np.array([[-0.8],[0.5],[-0.6],[0.9]])

a_outer_b = np.outer(a, b)
a_outer_a = np.outer(a, a)
b_outer_b = np.outer(b, b)

# Visualizations
plot_heatmap(a,'a')

plot_heatmap(b,'b')

plot_heatmap(a_outer_b,'a outer b')

plot_heatmap(a_outer_a,'a outer a')

plot_heatmap(b_outer_b,'b outer b')




















































































































































































































































































