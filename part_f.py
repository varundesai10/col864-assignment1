import numpy as np
from Partb import Sensor_model, Motion_model
from simulate import sense, move, lim, sensors
N = 25



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
    Path = []
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
                    Path.append((i, j-1))
                if ind == 1:
                    Path.append((i, j+1))
                if ind == 2:
                    Path.append((i-1, j))
                if ind == 3:
                    Path.append((i+1, j))
        eta = 0
        for i in range(30):
            for j in range(30):
                ProbE_X = Sensor_model(READINGS[l-1], (i, j))
                Sequence_probs[l, i, j] = Seq_prob_wo_obs[l-1, i,j]*ProbE_X
                eta+= Sequence_probs[l,i,j]
        Sequence_probs[l] = Sequence_probs[l]/eta
    return Path, Sequence_probs

Path, Sequence_probs = viterbi_algorithm(READINGS, N)
Path1 = np.array(Path)
Path = []
for i in range(N+1):
    Path.append(np.unravel_index(np.argmax(Sequence_probs[i], axis=None), Sequence_probs[N].shape))
print(Path)