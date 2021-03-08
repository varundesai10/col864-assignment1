from plot import plot_likelihood
from Partb import HMM_filter
from simulate import sense, move, lim, sensors
import numpy as np
import matplotlib.pyplot as plt 
import time

N = 25
curr_pos = None
POSITIONS = []
READINGS = []
for T in range(N):
    if(curr_pos is None):
        #curr_pos = (np.random.randint(lim[0], lim[1] + 1), np.random.randint(lim[2], lim[3] + 1))
        curr_pos = (12, 15)
    else:
        curr_pos = move(curr_pos)
    
    POSITIONS.append(curr_pos)
    print('moved to {}'.format(curr_pos))
    
    readings = sense(curr_pos, sensors)
    READINGS += [readings]
    print('readings = {}'.format(readings))

print('Final Print')
print('Positions = {} \nReadings = {}'.format(POSITIONS, READINGS))

bel = np.ones((30,30), dtype = np.float64)



for t in range(N):
    bel = HMM_filter(READINGS[t], bel)
    plot_likelihood(POSITIONS[t], bel)
