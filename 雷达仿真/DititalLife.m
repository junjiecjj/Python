clear all; close all; clc;

import time

figure(facecolor='k', newfig=False)
axes(position=[0,0,1,1], aspect='equal')
axis([0,400,0,400])
n = 2e4
i = arange(0, n, dtype='int')
sp1 = scatter(i, i, s=1, facecolor='w', edgecolor=None, alpha=0.4)
t = 0
x = mod(i, 100)
y = floor(i/100)
k = x/4 - 12.5
e = y/9 + 5
o = np.linalg.norm(np.row_stack((k,e)), axis=0)/9
for _ in range(100):
    t = t + pi/90
    q = x + 99 + tan(1./k) + o*k*(cos(e*9)/4 + cos(y/2))*sin(o*4 - t)
    c = o*e/30 - t/8
    sp1.xdata = (q*0.7*sin(c)) + 9*cos(y/19 + t) + 200
    sp1.ydata = 200 + (q/2*cos(c))
    plt.draw()
    time.sleep(0.05)