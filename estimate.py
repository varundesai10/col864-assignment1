from plot import plot_likelihood,plot_paths
from Partb import HMM_filter, Sensor_model, Motion_model
from simulate import sense, move, lim, sensors
import numpy as np
import matplotlib.pyplot as plt 
import time
import os

np.random.seed(1012)

BASE_IMAGE_PATH = os.path.abspath('./plots/')
PART_B_PATH = os.path.join(BASE_IMAGE_PATH, 'b/')
PART_C_PATH = os.path.join(BASE_IMAGE_PATH, 'c/')
PART_E_PATH = os.path.join(BASE_IMAGE_PATH, 'd/')
os.makedirs(BASE_IMAGE_PATH, exist_ok = True)
os.makedirs(PART_B_PATH, exist_ok = True)
os.makedirs(PART_C_PATH, exist_ok = True)
os.makedirs(PART_E_PATH, exist_ok = True)

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
#plot_likelihood((0,0), bel[0])
print(bel[0])

avg = 0

for t in range(1, N+1):
    
    bel[t] = HMM_filter(READINGS[t-1], bel[t-1])
    print(bel[t])
    ind = np.unravel_index(np.argmax(bel[t], axis=None), bel[t].shape)
    avg += d(ind, POSITIONS[t-1])
    plot_likelihood(POSITIONS[t-1], bel[t], estimated_position=ind, filename=os.path.join(PART_B_PATH, f'{t}.png'), title=f't = {t}')

print("Sum of manhattan distances = ", avg)
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

avg = 0
for t in range(1, N+1):
    
    ind = np.unravel_index(np.argmax(bel_smooth[t-1], axis=None), bel_smooth[t-1].shape)
    avg += d(ind, POSITIONS[t-1])
    plot_likelihood(POSITIONS[t-1], bel_smooth[t-1], estimated_position=ind, filename=os.path.join(PART_C_PATH, f'{t}.png'), title=f't = {t}')      

#into the future!!

bel = bel[N]
for k in range(25):
    temp = bel.copy()
    for i in range(30):
        for j in range(30):

            if(j-1 >= 0):
                temp[i,j] += bel[i, j-1] * Motion_model((i, j-1), (i,j))
            if(j + 1 < 30):
                temp[i,j] += bel[i, j+1] * Motion_model((i, j+1), (i,j))
            if(i-1 >= 0):
                temp[i,j] += bel[i-1, j] * Motion_model((i-1,j), (i,j))
            if(i+1 < 30):
                temp[i,j] += bel[i+1, j] * Motion_model((i+1, j), (i, j))
    bel = temp
    plot_likelihood(POSITIONS[N-1], bel, filename=os.path.join(PART_E_PATH, f'{k + 1}_future.png'), title=f'{k+1} steps into the future', error=False)
            
    
print("Sum of manhattan distances = ", avg)


def viterbi_algorithm(READINGS, N):
    Path_tracker_1 = np.frompyfunc(list, 0, 1)(np.empty((N+1,30,30), dtype=object))
    Sequence_probs = np.ones((N + 1, 30, 30), dtype=np.float64)
    Sequence_probs[0] = np.ones((30,30), dtype = np.float64)/900
    Seq_prob_wo_obs = np.zeros((N, 30, 30), dtype=np.float64)
    for l in range(1,N+1):

        for i in range(30):
            for j in range(30):
                list_of_prob =np.zeros(4, dtype = np.float64)
                if (j - 1 >= 0):
                    list_of_prob[0] = (Sequence_probs[l-1,i, j - 1] * Motion_model((i, j - 1), (i, j)))
                else:
                    list_of_prob[0] = -1
                if (j + 1 < 30):
                    list_of_prob[1] = (Sequence_probs[l-1,i, j + 1] * Motion_model((i, j + 1), (i, j)))
                else:
                    list_of_prob[1] = -1
                if (i - 1 >= 0):
                    list_of_prob[2] = (Sequence_probs[l-1,i-1, j] * Motion_model((i - 1, j), (i, j)))
                else:
                    list_of_prob[2] = -1

                if (i + 1 < 30):
                    list_of_prob[3] = (Sequence_probs[l-1,i+1, j] * Motion_model((i + 1, j), (i, j)))
                else:
                    list_of_prob[3] = -1
                ind = np.argmax(list_of_prob)
                Seq_prob_wo_obs[l-1, i,j] = list_of_prob[ind]
                if ind == 0:
                    list_= Path_tracker_1[l-1,i,j-1].copy()
                    list_.append((i,j-1))
                    Path_tracker_1[l,i,j] = list_
##check for the Path_Tracker
                if ind == 1:
                    list_ = Path_tracker_1[l-1,i, j +1].copy()
                    list_.append((i, j + 1))
                    Path_tracker_1[l,i, j] = list_


                if ind == 2:
                    list_ = Path_tracker_1[l-1,i-1, j].copy()
                    list_.append((i-1, j))
                    Path_tracker_1[l,i, j] = list_


                if ind == 3:

                    list_ = Path_tracker_1[l-1,i+1, j].copy()
                    list_.append((i+1, j))
                    Path_tracker_1[l,i, j] = list_


        eta = 0
        for i in range(30):
            for j in range(30):
                ProbE_X = Sensor_model(READINGS[l-1], (i, j))
                Sequence_probs[l, i, j] = Seq_prob_wo_obs[l-1, i,j]*ProbE_X
                eta+= Sequence_probs[l,i,j]
        Sequence_probs[l] = Sequence_probs[l]/eta
    return Path_tracker_1, Sequence_probs

PATH_T1,Sequence_probs=viterbi_algorithm(READINGS,N)
x,y = np.unravel_index(np.argmax(Sequence_probs[N], axis=None), Sequence_probs[N].shape)
print(len(PATH_T1[N,x,y]))
print(PATH_T1[N,x,y])
plot_paths(POSITIONS, PATH_T1[N, x, y], title="Result of the Viterbi Algorithm")

#plotting true and most likely path.