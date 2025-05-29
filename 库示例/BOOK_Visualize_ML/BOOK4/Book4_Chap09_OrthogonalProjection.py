



#%%  Bk4_Ch9_01.py

import numpy as np
import matplotlib.pyplot as plt

thetas = np.linspace(0, np.pi, 10)

x = np.array([[4], [3]])
fig, axes = plt.subplots(figsize = (10, 10))
for theta in thetas:
    v = np.array([[np.cos(theta)], [np.sin(theta)]])
    proj = (v.T@x)[0]
    # print(proj)
    plt.plot([-v[0]*6, v[0]*6], [-v[1]*6, v[1]*6])
    plt.plot([x[0,0], (v[0]*proj)[0]], [x[1,0], (v[1]*proj)[0]], color = 'k')
    plt.plot(v[0]*proj, v[1]*proj, color = 'k', marker = 'x')
    plt.quiver (0, 0, v[0], v[1], angles='xy', scale_units='xy',scale=1)
plt.plot(x[0],x[1], marker = 'x', color = 'r')
plt.axis('scaled')

#%% Bk4_Ch9_02.py
import numpy as np
import matplotlib.pyplot as plt

thetas = np.array([0, 15, 30, 45, 60, 75, 90, 120, 135])
x = np.array([[4], [3]])
i = 1
fig = plt.figure(figsize = (10, 10), constrained_layout = True)
for theta in thetas:
    theta = theta/180*np.pi
    ax = fig.add_subplot(3, 3, i)
    v1 = np.array([[np.cos(theta)], [np.sin(theta)]])
    proj = (v1.T@x)[0]
    print(proj)
    ax.plot([-v1[0]*6, v1[0]*6], [-v1[1]*6, v1[1]*6])
    # plt.plot([x[0], v1[0]*proj], [x[1], v1[1]*proj], color = 'k', linestyle = '--')
    ax.plot([x[0,0], (v[0]*proj)[0]], [x[1,0], (v[1]*proj)[0]], color = 'k', linestyle = '--')
    ax.plot(v1[0]*proj, v1[1]*proj, color = 'k', marker = 'x')

    ax.quiver (0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color = 'b')
    v2 = np.array([[-np.sin(theta)], [np.cos(theta)]])
    # proj = v2.T@x
    proj = (v2.T@x)[0]
    print(proj)
    ax.plot([-v2[0]*6, v2[0]*6], [-v2[1]*6, v2[1]*6])
    ax.plot([x[0,0], (v2[0]*proj)[0]], [x[1,0], (v2[1]*proj)[0]], color = 'k', linestyle = '--')
    ax.plot((v2[0]*proj)[0], (v2[1]*proj)[0], color = 'k', marker = 'x')

    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,color = 'r')

    ax.axhline(y = 0, color = 'k')
    ax.axvline(x = 0, color = 'k')
    ax.plot(x[0],x[1], marker = 'x', color = 'r')
    ax.quiver(0, 0, x[0],x[1], angles='xy', scale_units='xy', scale=1, color = 'k')

    ax.axis('scaled')
    ax.grid(linestyle='--', linewidth=0.25, color=[0.75,0.75,0.75])
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.set_xticks(np.linspace(-6,6,13))
    ax.set_yticks(np.linspace(-6,6,13))

    i = i + 1
































































































































































































































































































