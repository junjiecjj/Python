



import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

locs = ['upper right', 'lower left', 'center left', 'lower center', 'center', 'right']

x0, y0, width, height = 0.5, 0.5, 0.1, 0.4
x1, y1, width1, height1 = 0.5, 0.5, 0, 0

x = np.arange(0.1, 4, 0.1)
y = 1.0/x

fig = plt.figure(figsize=(10, 10))

idx = 1
for i in range(0, 2):
    for j in range(0, 3):
        ax = fig.add_subplot(3, 2, idx)
        ax.plot(x, y, label=r'$\frac{1}{x}$')
        ax.legend(loc=locs[idx-1], bbox_to_anchor=(x0, y0, width, height), edgecolor='g', fontsize='large', framealpha=0.5, borderaxespad=0)
        ax.add_patch( patches.Rectangle((x0, y0), width, height, color='r', fill=False, transform=ax.transAxes) )
        ax.text(0.6, 0.2, s="loc = '{}'".format(locs[idx-1]),
        transform=ax.transAxes)
        idx += 1

fig1 = plt.figure(figsize=(10, 10))

idx1 = 1
for i in range(0, 2):
    for j in range(0, 3):
        ax1 = fig1.add_subplot(3, 2, idx1)
        ax1.plot(x, y, label=r'$\frac{1}{x}$')
        ax1.legend(loc=locs[idx1-1], bbox_to_anchor=(x1, y1, width1, height1), edgecolor='g', fontsize='large', framealpha=0.5, borderaxespad=0)
        ax1.add_patch( patches.Rectangle((x1, y1), width1, height1, color='r', fill=False, transform=ax1.transAxes) )
        ax1.text(0.6, 0.2, s="loc = '{}'".format(locs[idx1-1]),
        transform=ax1.transAxes)
        idx1 += 1

plt.show()
