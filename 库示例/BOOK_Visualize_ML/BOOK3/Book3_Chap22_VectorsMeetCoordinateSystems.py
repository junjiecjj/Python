



#%% Bk3_Ch22_1

import numpy as np
import matplotlib.pyplot as plt

# draw vectors starting from origin

def draw_vector(vector,RBG,label):
    array = np.array([[0, 0, vector[0], vector[1]]])
    X, Y, U, V = zip(*array)
    plt.quiver(X, Y, U, V,angles='xy', scale_units='xy',scale=1,color = RBG)

    # add labels to the sample data

    label = label + f" ({vector[0]},{vector[1]})"

    plt.annotate(label, # text
                 (vector[0],vector[1]), # point to label
                 textcoords="offset points",
                 xytext=(0,10),
                 # distance from text to points (x,y)
                 ha='center')
                 # horizontal alignment center

# define one vector
a = np.array([4,3])
i = np.array([1,0])
j = np.array([0,1])

fig, ax = plt.subplots()

draw_vector(4*i, np.array([0,112,192])/255, '4i')
draw_vector(3*j, np.array([255,0,0])/255, '3j')

draw_vector(i, np.array([0,112,192])/255, 'i')
draw_vector(j, np.array([255,0,0])/255, 'j')
draw_vector(a,np.array([146,208,80])/255, 'a')

plt.xlabel('$x$')
plt.ylabel('$y$')

plt.axis('scaled')
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.show()
plt.close()


#%% Bk3_Ch22_2

import numpy as np
import matplotlib.pyplot as plt

# draw vectors starting from origin
def draw_vector(vector,RBG,label):
    array = np.array([[0, 0, vector[0], vector[1]]])
    X, Y, U, V = zip(*array)
    plt.quiver(X, Y, U, V,angles='xy', scale_units='xy',scale=1,color = RBG)
    # add labels to the sample data
    label = label + f" ({vector[0]},{vector[1]})"
    plt.annotate(label, # text
                 (vector[0],vector[1]), # point to label
                 textcoords="offset points",
                 xytext=(0,10),
                 # distance from text to points (x,y)
                 ha='center')
                 # horizontal alignment center

# define two vectors
a = np.array([4,1])
b = np.array([1,3])

# addition of a and b
fig, ax = plt.subplots()
draw_vector(a, np.array([0,112,192])/255, 'a')
draw_vector(b, np.array([255,0,0])/255, 'b')
draw_vector(a + b,np.array([146,208,80])/255, 'a + b')

plt.xlabel('$x$')
plt.ylabel('$y$')

plt.axis('scaled')
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.show()
plt.close()

# subtraction, a - b
fig, ax = plt.subplots()

draw_vector(a, np.array([0,112,192])/255, 'a')
draw_vector(b, np.array([255,0,0])/255, 'b')
draw_vector(a - b,np.array([146,208,80])/255, 'a - b')

plt.xlabel('$x$')
plt.ylabel('$y$')

plt.axis('scaled')
ax.set_xlim([0, 5])
ax.set_ylim([-2, 3])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.show()
plt.close()

a = np.array([2,2])

# scalar multiplication
fig, ax = plt.subplots()

draw_vector(2*a, np.array([146,208,80])/255, '2*a')
draw_vector(a, np.array([0,112,192])/255, 'a')
draw_vector(0.5*a, np.array([255,0,0])/255, '0.5*a')


plt.xlabel('$x$')
plt.ylabel('$y$')

plt.axis('scaled')
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.show()
plt.close()


#%% Bk3_Ch22_3

import numpy as np
a = [4,1]
b = [1,3]

# L2 norms
a_norm = np.linalg.norm(a)
b_norm = np.linalg.norm(b)

# dot product
a_dot_b = np.dot(a, b)

# cosine result
cos_result = a_dot_b/a_norm/b_norm

# radian to degree
# np.arccos(cos_result)*180/np.pi
angle = np.degrees(np.arccos(cos_result))

print(angle)


#%% Bk3_Ch22_4

import numpy as np
import matplotlib.pyplot as plt

def draw_vector(vector, RBG, label,zdir):
    array = np.array([[0, 0, 0, vector[0], vector[1], vector[2]]])
    X, Y, Z, U, V, W = zip(*array)
    plt.quiver(X, Y, Z, U, V, W, normalize = False, color = RBG, arrow_length_ratio=0.1)

    label = label + ' (%d, %d, %d)' %(vector[0], vector[1], vector[2])

    ax.text(vector[0], vector[1], vector[2], label, zdir, verticalalignment='center')

# define one vector
c = np.array([4, 3, 5])
a = np.array([4, 3, 0])
i = np.array([1, 0, 0])
j = np.array([0, 1, 0])
k = np.array([0, 0, 1])

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (10, 10))

draw_vector(a,np.array([0,0,0])/255, 'a',a)
draw_vector(c,np.array([0,0,0]), 'c',c)

draw_vector(i, np.array([0,112,192])/255,  'i',(1,0,0))
draw_vector(j, np.array([255,0,0])/255,    'j',(0,1,0))
draw_vector(k, np.array([146,208,80])/255, 'k',(0,0,1))

# plt.show()
ax.set_proj_type('ortho')

ax.set_xlim(0,5)
ax.set_ylim(0,5)
ax.set_zlim(0,5)
ax.spines['left'].set_position('zero')

plt.tight_layout()
ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{z}$')

ax.view_init(azim=60, elev=20)
# ax.view_init(azim=30, elev=20)
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

plt.show()
plt.close()




























































































































































































































































































