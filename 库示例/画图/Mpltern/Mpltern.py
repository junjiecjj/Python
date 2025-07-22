#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:01:25 2024

@author: jack
"""

#%%>>>>>>>>>>>>>>>>>>>>>   Scatter
import numpy as np

import matplotlib.pyplot as plt
import mpltern


np.random.seed(seed=19)
t0, l0, r0 = np.random.dirichlet(alpha=(2.0, 2.0, 2.0), size=100).T

np.random.seed(seed=68)
t1, l1, r1 = np.random.dirichlet(alpha=(2.0, 2.0, 2.0), size=100).T

dt = t1 - t0
dl = l1 - l0
dr = r1 - r0

length = np.sqrt(dt**2 + dl**2 + dr**2)

fig = plt.figure(figsize=(10.8, 4.8))
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)

ax = fig.add_subplot(1, 2, 1, projection="ternary")
pc = ax.scatter(t0, l0, r0)
pc = ax.scatter(t1, l1, r1)

ax = fig.add_subplot(1, 2, 2, projection="ternary")
pc = ax.scatter(t0, l0, r0, c=length)

cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
colorbar = fig.colorbar(pc, cax=cax)
colorbar.set_label("Length", rotation=270, va="baseline")

plt.show()
plt.close('all')

#%%>>>>>>>>>>>>>>>>>>>>>   Quiver
import numpy as np

import matplotlib.pyplot as plt
from mpltern.datasets import get_triangular_grid


t, l, r = get_triangular_grid()

# Arrows. The sum of the three must be zero.
dt = 1.0 / 3.0 - t
dl = 1.0 / 3.0 - l
dr = 1.0 / 3.0 - r

length = np.sqrt(dt ** 2 + dl ** 2 + dr ** 2)

fig = plt.figure(figsize=(10.8, 4.8))
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)

ax = fig.add_subplot(121, projection='ternary')
pc = ax.quiver(t, l, r, dt, dl, dr)

ax = fig.add_subplot(122, projection='ternary')
pc = ax.quiver(t, l, r, dt, dl, dr, length)

cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
colorbar = fig.colorbar(pc, cax=cax)
colorbar.set_label('Length', rotation=270, va='baseline')

plt.show()
plt.close('all')


#%%>>>>>>>>>>>>>>>>>>>>> Dirichlet distribution
import matplotlib.pyplot as plt
from mpltern.datasets import get_dirichlet_pdfs

fig = plt.figure(figsize=(10.8, 8.8))
fig.subplots_adjust(
    left=0.1,
    right=0.9,
    bottom=0.1,
    top=0.9,
    wspace=0.5,
    hspace=0.5,
)

alphas = ((1.5, 1.5, 1.5), (5.0, 5.0, 5.0), (1.0, 2.0, 2.0), (2.0, 4.0, 8.0))
for i, alpha in enumerate(alphas):
    ax = fig.add_subplot(2, 2, i + 1, projection="ternary")
    t, l, r, v = get_dirichlet_pdfs(n=61, alpha=alpha)
    cmap = "Blues"
    shading = "gouraud"
    cs = ax.tripcolor(t, l, r, v, cmap=cmap, shading=shading, rasterized=True)
    ax.tricontour(t, l, r, v, colors="k", linewidths=0.5)

    ax.set_tlabel("$x_1$")
    ax.set_llabel("$x_2$")
    ax.set_rlabel("$x_3$")

    ax.taxis.set_label_position("tick1")
    ax.laxis.set_label_position("tick1")
    ax.raxis.set_label_position("tick1")

    ax.set_title("${\\mathbf{\\alpha}}$ = " + str(alpha))

    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
    colorbar = fig.colorbar(cs, cax=cax)
    colorbar.set_label("PDF", rotation=270, va="baseline")

plt.show()
plt.close('all')

#%%>>>>>>>>>>>>>>>>>>>>> Tick locators
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import mpltern

ax = plt.subplot(projection="ternary")

np.random.seed(19680801)
t, l, r = np.random.dirichlet(alpha=(2.0, 4.0, 8.0), size=128).T

ax.scatter(t, l, r, s=64.0, c="C1", edgecolors="k", alpha=0.6)

ax.set_tlabel("$x_1$")
ax.set_llabel("$x_2$")
ax.set_rlabel("$x_3$")

ax.taxis.set_major_locator(MultipleLocator(0.25))
ax.laxis.set_major_locator(MultipleLocator(0.20))
ax.raxis.set_major_locator(MultipleLocator(0.10))

ax.laxis.set_minor_locator(MultipleLocator(0.1))
ax.raxis.set_minor_locator(AutoMinorLocator(5))

ax.grid(axis='t')
ax.grid(axis='l', which='minor', linestyle='--')
ax.grid(axis='r', which='both', linestyle=':')

plt.show()
plt.close('all')

#%%>>>>>>>>>>>>>>>>>>>>> Colored axes
import matplotlib.pyplot as plt
from mpltern.datasets import get_spiral


fig = plt.figure()
ax = fig.add_subplot(projection='ternary')

ax.plot(*get_spiral(), color='k')

ax.set_tlabel('Top')
ax.set_llabel('Left')
ax.set_rlabel('Right')

ax.grid()

# Color ticks, grids, tick-labels
ax.taxis.set_tick_params(tick2On=True, colors='C0', grid_color='C0')
ax.laxis.set_tick_params(tick2On=True, colors='C1', grid_color='C1')
ax.raxis.set_tick_params(tick2On=True, colors='C2', grid_color='C2')

# Color labels
ax.taxis.label.set_color('C0')
ax.laxis.label.set_color('C1')
ax.raxis.label.set_color('C2')

# Color spines
ax.spines['tside'].set_color('C0')
ax.spines['lside'].set_color('C1')
ax.spines['rside'].set_color('C2')

plt.show()
plt.close('all')

#%%>>>>>>>>>>>>>>>>>>>>> Contour
import numpy as np

import matplotlib.pyplot as plt
from mpltern.datasets import get_shanon_entropies

t, l, r, v = get_shanon_entropies()

fig = plt.figure(figsize=(10.8, 4.8))
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)

# These values are for controlling the color-bar scale, and here they are
# explicitly given just to make the same color-bar scale for all the plots.
# In general, you may not need to explicitly specify them.
vmin = 0.0
vmax = 1.2
levels = np.linspace(vmin, vmax, 7)

ax = fig.add_subplot(1, 2, 1, projection='ternary')
cs = ax.tricontour(t, l, r, v, levels=levels)
ax.clabel(cs)
ax.set_title("tricontour")

cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
colorbar = fig.colorbar(cs, cax=cax)
colorbar.set_label('Entropy', rotation=270, va='baseline')

ax = fig.add_subplot(1, 2, 2, projection='ternary')
cs = ax.tricontourf(t, l, r, v, levels=levels)
ax.set_title("tricontourf")

cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
colorbar = fig.colorbar(cs, cax=cax)
colorbar.set_label('Entropy', rotation=270, va='baseline')

plt.show()
plt.close('all')

#%%>>>>>>>>>>>>>>>>>>>>> Triangle rotation
import matplotlib.pyplot as plt
from mpltern.datasets import get_spiral


t, l, r = get_spiral()

fig = plt.figure(figsize=(10.8, 8.8))
fig.subplots_adjust(
    left=0.1,
    right=0.9,
    hspace=0.75,
)

rotations = range(0, 360, 90)
for i, rotation in enumerate(rotations):
    ax = fig.add_subplot(2, 2, i + 1, projection='ternary', rotation=rotation)

    ax.plot(t, l, r)

    ax.set_tlabel('Top')
    ax.set_llabel('Left')
    ax.set_rlabel('Right')

    ax.set_title(f"rotation={rotation}")

plt.show()
plt.close('all')

#%%>>>>>>>>>>>>>>>>>>>>> Evolutionary game theory
import matplotlib.pyplot as plt
import numpy as np
from mpltern.datasets import get_triangular_grid

x = np.array(get_triangular_grid(25))

payoff_matrix = [
    [0.0, -1.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 1.0, 0.0],
]

fitness = payoff_matrix @ x
d = (fitness - np.sum(fitness * x, axis=0)) * x
norm = np.linalg.norm(d, axis=0)

ax = plt.subplot(projection="ternary")

ax.tripcolor(*x, norm, cmap="turbo", shading="gouraud", rasterized=True)

ax.quiver(*x, *d, scale=5, clip_on=False)

ax.set_tlabel("$x_0$")
ax.set_llabel("$x_1$")
ax.set_rlabel("$x_2$")

plt.show()
plt.close('all')

#%%>>>>>>>>>>>>>>>>>>>>> Axis and Tick

# Triangle rotation
import matplotlib.pyplot as plt
from mpltern.datasets import get_spiral


t, l, r = get_spiral()

fig = plt.figure(figsize=(10.8, 8.8))
fig.subplots_adjust(
    left=0.1,
    right=0.9,
    hspace=0.75,
)

rotations = range(0, 360, 90)
for i, rotation in enumerate(rotations):
    ax = fig.add_subplot(2, 2, i + 1, projection='ternary', rotation=rotation)

    ax.plot(t, l, r)

    ax.set_tlabel('Top')
    ax.set_llabel('Left')
    ax.set_rlabel('Right')

    ax.set_title(f"rotation={rotation}")

plt.show()
plt.close('all')

# Ticks
import matplotlib.pyplot as plt
import mpltern  # noqa: F401

ax = plt.subplot(projection="ternary", ternary_sum=100.0)

x0 = 0.0
x1 = 100.0

ax.plot([x0, x1], [0.5, 0.5], transform=ax.get_taxis_transform())
ax.plot([x0, x1], [0.5, 0.5], transform=ax.get_laxis_transform())
ax.plot([x0, x1], [0.5, 0.5], transform=ax.get_raxis_transform())

y = [0.4, 0.6]

ax.fill_betweenx(y, x0, x1, alpha=0.2, transform=ax.get_taxis_transform())
ax.fill_betweenx(y, x0, x1, alpha=0.2, transform=ax.get_laxis_transform())
ax.fill_betweenx(y, x0, x1, alpha=0.2, transform=ax.get_raxis_transform())

plt.show()
plt.close('all')


# Axis-label rotation
import matplotlib.pyplot as plt
from mpltern.datasets import get_spiral


pad = 42

t, l, r = get_spiral()

fig = plt.figure(figsize=(10.8, 4.8))
fig.subplots_adjust(wspace=0.3)

modes = ['axis', 'horizontal']
for i, mode in enumerate(modes):
    ax = fig.add_subplot(1, 2, i + 1, projection='ternary')

    ax.plot(t, l, r)

    ax.set_tlabel('Top')
    ax.set_llabel('Left')
    ax.set_rlabel('Right')

    ax.taxis.set_label_rotation_mode(mode)
    ax.laxis.set_label_rotation_mode(mode)
    ax.raxis.set_label_rotation_mode(mode)

    ax.set_title("label_rotation_mode='{}'".format(mode), pad=pad)

plt.show()
plt.close('all')


# Tick rotation
import matplotlib.pyplot as plt
import mpltern

fig = plt.figure(figsize=(10.8, 4.8))
fig.subplots_adjust(left=0.075, right=0.925, wspace=0.25)

for i, labelrotation in enumerate(["tick", "axis", "horizontal"]):
    ax = fig.add_subplot(1, 3, i + 1, projection='ternary')
    ax.tick_params(labelrotation=labelrotation)
    ax.set_title(f"labelrotation='{labelrotation}'", pad=36)
plt.show()
plt.close('all')

# Tick direction
import matplotlib.pyplot as plt
import mpltern

fig = plt.figure(figsize=(10.8, 4.8))
fig.subplots_adjust(left=0.075, right=0.925, wspace=0.25)

for i, direction in enumerate(["in", "out", "inout"]):
    ax = fig.add_subplot(1, 3, i + 1, projection='ternary')
    ax.tick_params(direction=direction)
    ax.set_title(f"ax.tick_params(direction='{direction}')", pad=36)
plt.show()
plt.close('all')

# Tick position
import matplotlib.pyplot as plt
from mpltern.datasets import get_spiral


t, l, r = get_spiral()

fig = plt.figure(figsize=(10.8, 4.8))
fig.subplots_adjust(wspace=0.3)

positions = ['tick1', 'tick2']
for i, position in enumerate(positions):
    ax = fig.add_subplot(1, 2, i + 1, projection='ternary')

    ax.plot(t, l, r)

    ax.set_tlabel('Top')
    ax.set_llabel('Left')
    ax.set_rlabel('Right')

    ax.taxis.set_ticks_position(position)
    ax.laxis.set_ticks_position(position)
    ax.raxis.set_ticks_position(position)

    ax.taxis.set_label_position(position)
    ax.laxis.set_label_position(position)
    ax.raxis.set_label_position(position)

    ax.set_title(f"position='{position}'", pad=42)
plt.show()
plt.close('all')


# Tick formatters
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import mpltern

ternary_sum = 100.0

ax = plt.subplot(projection="ternary", ternary_sum=ternary_sum)

np.random.seed(19680801)
t, l, r = ternary_sum * np.random.dirichlet(alpha=(2.0, 4.0, 8.0), size=128).T

ax.scatter(t, l, r, s=64.0, c="none", edgecolors="C0")

ax.set_tlabel("$x_1$")
ax.set_llabel("$x_2$")
ax.set_rlabel("$x_3$")

ax.taxis.set_major_formatter(PercentFormatter())
ax.laxis.set_major_formatter(PercentFormatter())
ax.raxis.set_major_formatter(PercentFormatter())

plt.show()
plt.close('all')

#%%>>>>>>>>>>>>>>>>>>>>>





#%%>>>>>>>>>>>>>>>>>>>>>





#%%>>>>>>>>>>>>>>>>>>>>>





#%%>>>>>>>>>>>>>>>>>>>>>





#%%>>>>>>>>>>>>>>>>>>>>>





#%%>>>>>>>>>>>>>>>>>>>>>





#%%>>>>>>>>>>>>>>>>>>>>>





#%%>>>>>>>>>>>>>>>>>>>>>







