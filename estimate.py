from plot import plot_likelihood
from Partb import HMM_filter, Sensor_model, Motion_model
from simulate import sense, move, lim, sensors
import numpy as np
import matplotlib.pyplot as plt 
import time
import os

BASE_IMAGE_PATH = os.path.abspath('./plots/')
PART_B_PATH = os.path.join(BASE_IMAGE_PATH, 'b/')
PART_C_PATH = os.path.join(BASE_IMAGE_PATH, 'c/')

os.makedirs(BASE_IMAGE_PATH, exist_ok = True)
os.makedirs(PART_B_PATH, exist_ok = True)
os.makedirs(PART_C_PATH, exist_ok = True)

def d(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

N = 25
curr_pos = None
POSITIONS = []
READINGS = []
for T in range(N):
    if(curr_pos is None):
        curr_pos = (np.random.randint(lim[0], lim[1] + 1), np.random.randint(lim[2], lim[3] + 1))
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

bel = np.ones((N+1, 30,30), dtype = np.float64)
bel[0] = bel[0]*(1/900)
print(np.max(bel[0]), np.min(bel[0]))
plot_likelihood((0,0), bel[0])
print(bel[0])
for t in range(1, N+1):
    
    bel[t] = HMM_filter(READINGS[t-1], bel[t-1])
    print(bel[t])
    ind = np.unravel_index(np.argmax(bel[t], axis=None), bel[t].shape)
    plot_likelihood(POSITIONS[t-1], bel[t], estimated_position=ind, filename=os.path.join(PART_B_PATH, f'{t}.png'), title=f't = {t}')

#backward loop!

bel_smooth = np.zeros((N, 30, 30), dtype=np.float64)
backward = np.ones((30,30), dtype=np.float64) #belief that propogates backward

bel_smooth[N-1] = bel[N]

t=N-1
while(t>0):
    for i in range(30):
        for j in range(30):
            if i + 1 < 30:
                bel_smooth[t-1, i, j] += np.nan_to_num(Sensor_model(READINGS[t], (i + 1, j))*backward[i+1, j]*Motion_model((i, j), (i+1,j)))
            if i - 1 >= 0:
                bel_smooth[t-1, i, j] += np.nan_to_num(Sensor_model(READINGS[t], (i - 1, j))*backward[i-1, j]*Motion_model((i, j), (i-1,j)))
            if j + 1 < 30:
                bel_smooth[t-1, i, j] += np.nan_to_num(Sensor_model(READINGS[t], (i, j+1))*backward[i, j+1]*Motion_model((i, j), (i,j+1)))
            if j - 1 >= 0:
                bel_smooth[t-1, i, j] += np.nan_to_num(Sensor_model(READINGS[t], (i, j-1))*backward[i, j-1]*Motion_model((i, j), (i,j-1)))
    
    backward = np.nan_to_num(bel_smooth[t-1])
    bel_smooth[t-1] = np.nan_to_num(np.dot(bel[t], backward))
    bel_smooth[t-1] /= np.sum(bel_smooth[t-1])
    bel_smooth[t-1] = np.nan_to_num(bel_smooth[t-1])
    print(f'bel_smooth {t} = ', bel_smooth[t-1])
    t-=1

for t in range(1, N+1):
    
    ind = np.unravel_index(np.argmax(bel_smooth[t-1], axis=None), bel_smooth[t-1].shape)
    plot_likelihood(POSITIONS[t-1], bel_smooth[t-1], estimated_position=ind, filename=os.path.join(PART_C_PATH, f'{t}.png'), title=f't = {t}')      

