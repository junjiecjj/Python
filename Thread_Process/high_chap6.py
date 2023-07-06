
#!/usr/bin/env python3
#! -*- coding: utf-8 -*-
import time

grid_shape = (1024,1024)

def envolve(grid,dt,out,D=1.0):
    xmax, ymax = grid_shape
    for i in range(xmax):
        for j in range(ymax):
            grid_xx = grid[(i+1)%xmax][j]+grid[(i-1)%xmax][j]-2.0*grid[i][j]
            grid_yy = grid[i][(j+1)%ymax]+grid[i][(j-1)%ymax]-2.0*grid[i][j]
            out[i][j] = grid[i][j]+D*(grid_xx+grid_yy)*dt

def run_experiment(num_iterations):
    xmax,ymax = grid_shape
    next_grid = [[0.0,]*ymax for x in range(xmax)]
    grid = [[0.0,]*ymax for x in range(xmax)]

    block_low = int(grid_shape[0]*.4)
    block_high= int(grid_shape[0]*.5)
    for i in range(block_low,block_high):
        for j in range(block_low,block_high):
            grid[i][j] = 0.005
    start = time.time()
    for i in range(num_iterations):
        envolve(grid,0.1,next_grid)
        grid, next_grid = next_grid , grid
    return time.time()-start

time = run_experiment(10)
print("Time is % f sec."% time)
