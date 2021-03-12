import numpy as np
from Partb import HMM_filter, Motion_model
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

Bel = np.ones((30,30), dtype = np.float64)*(1/900)

bel = np.ones((N + 1, 30, 30), dtype=np.float64)
bel[0] = bel[0] * (1 / 900)
for t in range(1, N + 1):
    bel[t] = HMM_filter(READINGS[t - 1], bel[t - 1])

#Prediction_function
def Prediction(Bel, no_of_time_steps_in_the_future):
    Pred_t_1 = Bel
    T = no_of_time_steps_in_the_future
    Pred_t = np.zeros((30,30), dtype = np.float64)
    for l in range(T):
        for i in range(30):
            for j in range(30):
                if (j - 1 >= 0):
                    Pred_t[i, j] += Pred_t_1[i, j - 1] * Motion_model((i, j - 1), (i, j))
                if (j + 1 < 30):
                    Pred_t[i, j] += Pred_t_1[i, j + 1] * Motion_model((i, j + 1), (i, j))
                if (i - 1 >= 0):
                    Pred_t[i, j] += Pred_t_1[i - 1, j] * Motion_model((i - 1, j), (i, j))
                if (i + 1 < 30):
                    Pred_t[i, j] += Pred_t_1[i + 1, j] * Motion_model((i + 1, j), (i, j))
        Pred_t_1 = Pred_t
    return Pred_t

Pred = Prediction(bel[N], 1)
print(np.unravel_index(np.argmax(Pred, axis=None), Pred.shape))

