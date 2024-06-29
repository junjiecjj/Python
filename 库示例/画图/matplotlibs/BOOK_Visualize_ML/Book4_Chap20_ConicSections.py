


#%% Bk4_Ch20_01.py

import numpy as np
import matplotlib.pyplot as plt

alphas = np.linspace(0, 2*np.pi, 100)

# unit circle
r = np.sqrt(1.0)

z1 = r*np.cos(alphas)
z2 = r*np.sin(alphas)

Z = np.array([z1, z2]).T # data of unit circle

# scale
S = np.array([[2, 0],
              [0, 0.5]])

thetas = np.array([0, 30, 45, 60, 90, 120])

for theta in thetas:

    # rotate
    print('==== Rotate ====')
    print(theta)
    theta = theta/180*np.pi
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    # translate
    c = np.array([2, 1])
    X = Z@S@R.T + c;

    Q = R@np.linalg.inv(S)@np.linalg.inv(S)@R.T
    # print('==== Q ====')
    # print(Q)
    LAMBDA, V = np.linalg.eig(Q)
    # print('==== LAMBDA ====')
    # print(LAMBDA)
    # print('==== V ====')
    # print(V)

    x1 = X[:,0]
    x2 = X[:,1]

    fig, ax = plt.subplots(1)
    ax.plot(z1, z2, 'b') # plot the unit circle
    ax.plot(x1, x2, 'r') # plot the transformed shape
    ax.plot(c[0],c[1],'xk') # plot the center

    ax.quiver(0,0,1,0,color = 'b',angles='xy', scale_units='xy',scale=1)
    ax.quiver(0,0,0,1,color = 'b',angles='xy', scale_units='xy',scale=1)
    ax.quiver(0,0,-1,0,color = 'b',angles='xy', scale_units='xy',scale=1)
    ax.quiver(0,0,0,-1,color = 'b',angles='xy', scale_units='xy',scale=1)

    ax.quiver(0,0,c[0],c[1],color = 'k',angles='xy', scale_units='xy',scale=1)

    ax.quiver(c[0],c[1],
              V[0,0]/np.sqrt(LAMBDA[0]),
              V[1,0]/np.sqrt(LAMBDA[0]),color = 'r',
              angles='xy', scale_units='xy',scale=1)

    ax.quiver(c[0],c[1],
              V[0,1]/np.sqrt(LAMBDA[1]),
              V[1,1]/np.sqrt(LAMBDA[1]),color = 'r',
              angles='xy', scale_units='xy',scale=1)

    ax.quiver(c[0],c[1],
              -V[0,0]/np.sqrt(LAMBDA[0]),
              -V[1,0]/np.sqrt(LAMBDA[0]),color = 'r',
              angles='xy', scale_units='xy',scale=1)

    ax.quiver(c[0],c[1],
              -V[0,1]/np.sqrt(LAMBDA[1]),
              -V[1,1]/np.sqrt(LAMBDA[1]),color = 'r',
              angles='xy', scale_units='xy',scale=1)

    plt.axvline(x=0, color= 'k', zorder=0)
    plt.axhline(y=0, color= 'k', zorder=0)


    ax.set_aspect(1)
    plt.xlim(-2,4)
    plt.ylim(-2,4)
    plt.grid(linestyle='--')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()



#%% Bk4_Ch20_02.py

import numpy as np
import matplotlib.pyplot as plt

alphas = np.linspace(0, 2*np.pi, 100)

# unit circle
r = np.sqrt(1.0)

z1 = r*1/np.cos(alphas)
z2 = r*np.tan(alphas)

Z = np.array([z1, z2]).T # data of unit circle

# scale
S = np.array([[1, 0],
              [0, 1]])

thetas = np.array([0, 30, 45, 60, 90, 120])

for theta in thetas:
    # rotate
    # print('==== Rotate ====')
    # print(theta)
    theta = theta/180*np.pi
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    X = Z@S@R.T;

    x1 = X[:,0]
    x2 = X[:,1]

    fig, ax = plt.subplots(1)
    ax.plot(z1, z2, 'b') # plot the unit circle
    ax.plot(x1, x2, 'r') # plot the transformed shape

    plt.axvline(x=0, color= 'k', zorder=0)
    plt.axhline(y=0, color= 'k', zorder=0)
    ax.set_aspect(1)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.grid(linestyle='--')

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()

#%% Bk4_Ch20_03.py

import numpy as np
import matplotlib.pyplot as plt

a = 1.5
b = 1

x1 = np.linspace(-3,3,200)
x2 = np.linspace(-3,3,200)
xx1,xx2 = np.meshgrid(x1,x2)

fig, ax = plt.subplots()

theta_array = np.linspace(0,2*np.pi,100)

plt.plot(a*np.cos(b*np.sin(theta)),b*np.sin(b*np.sin(theta)),color = 'k')

colors = plt.cm.RdYlBu(np.linspace(0,1,len(theta_array)))

for i in range(len(theta_array)):

    theta = theta_array[i]

    p1 = a*np.cos(theta)
    p2 = b*np.sin(theta)

    tangent = p1*xx1/a**2 + p2*xx2/b**2 - p1**2/a**2 - p2**2/b**2

    colors_i = colors[int(i),:]

    ax.contour(xx1,xx2,tangent, levels = [0], colors = [colors_i])

plt.axis('scaled')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axvline(x=0,color = 'k')
ax.axhline(y=0,color = 'k')


































































































































































































































































































