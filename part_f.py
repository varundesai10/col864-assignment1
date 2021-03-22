import numpy as np
from Partb import Sensor_model, Motion_model
from simulate import sense, move, lim, sensors



N = 30
curr_pos = None
POSITIONS = []
READINGS = []
for T in range(N):
    if (curr_pos is None):
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

def viterbi_algorithm(READINGS, N):
    ##check for the Path_Tracker
    Path_tracker_1 = np.empty((30,30), dtype = object)
    Path_tracker_1.fill([])
    Path_tracker_2 = np.empty((30,30), dtype = object)
    Path_tracker_2.fill([])
    Sequence_probs = np.ones((N + 1, 30, 30), dtype=np.float64)
    Sequence_probs[0] = np.ones((30,30), dtype = np.float64)/900
    Seq_prob_wo_obs = np.zeros((N, 30, 30), dtype=np.float64)
    for l in range(1,N+1):
        Path_tracker_1 = Path_tracker_2
        print(l)
        print(len(Path_tracker_1[1,1]))
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
                    list_= Path_tracker_1[i,j-1]
                    list_.append((i,j-1))
                    Path_tracker_2[i,j] = list_
##check for the Path_Tracker
                if ind == 1:
                    list_ = Path_tracker_1[i, j +1]
                    list_.append((i, j + 1))
                    Path_tracker_2[i, j] = list_


                if ind == 2:
                    list_ = Path_tracker_1[i-1, j]
                    list_.append((i-1, j))
                    Path_tracker_2[i, j] = list_


                if ind == 3:

                    list_ = Path_tracker_1[i+1, j]
                    list_.append((i+1, j))
                    Path_tracker_2[i, j] = list_


        eta = 0
        for i in range(30):
            for j in range(30):
                ProbE_X = Sensor_model(READINGS[l-1], (i, j))
                Sequence_probs[l, i, j] = Seq_prob_wo_obs[l-1, i,j]*ProbE_X
                eta+= Sequence_probs[l,i,j]
        Sequence_probs[l] = Sequence_probs[l]/eta
    return Path_tracker_1, Path_tracker_2, Sequence_probs

PATH_T1,PATH_T2,Sequence_probs=viterbi_algorithm(READINGS,N)
