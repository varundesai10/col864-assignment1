import numpy as np
from simulate import move, sense
lim = [0,29,0,29]
move_probabilies = [0.4, 0.1, 0.2, 0.3] #up, down, left, right
move_dict = {0:lambda x, y: (x, min(y+1, lim[3])) , 1:lambda x, y: (x, max(y-1, lim[2])), 2:lambda x, y: (max(x-1, lim[0]),y), 3:lambda x, y: (min(x+1, lim[1]), y)}

sensors = [(8, 15),(15, 15),(22, 15),(15, 22)]
p_sensor = [0.9, 0.8, 0.7, 0.6, 0.5]

curr_pos = (0,0)

<<<<<<< HEAD
Bel = np.ones((30,30), dtype = np.float64)*(1/900)
=======
Bel = np.ones((30,30))*(1/900)
>>>>>>> rocktim


def Sensor_model(Obs_E, X):
    x_pos, y_pos =X

    probE_given_X = 1

    for i, sensor in enumerate(sensors):
        x_sen, y_sen = sensor
        a = int(max(np.abs(x_pos-x_sen), np.abs(np.abs(y_pos-y_sen))))
        
        if a < len(p_sensor) and Obs_E[i] == 1:
            probE_given_X = probE_given_X*p_sensor[a]
        
        elif a < len(p_sensor) and Obs_E[i] == 0:
            probE_given_X = probE_given_X *(1 - p_sensor[a])
        
        elif a >= len(p_sensor) and Obs_E[i] == 1:
            probE_given_X = probE_given_X*0
        
        elif a >= len(p_sensor) and Obs_E[i] == 0:
            probE_given_X = probE_given_X*1

    return probE_given_X

def Motion_model(curr_pos, next_position):
    x, y = curr_pos
    x_next, y_next = next_position
    
    #invalid positions
    if x_next >=30 or y_next >=30 or x_next <0 or y_next<0 or x>=30 or y>=30 or x<0 or y<0:
        return 0.0
    
    a_x = x_next - x
    a_y = y_next - y
    
    #no diagonal movement
    if a_x !=0 and a_y != 0:
        return 0.0

    #normal movements
    if a_x == 1:
        return 0.3
    elif a_x == -1:
        return 0.2
    elif a_y == 1:
        return 0.4
    elif a_y == -1:
        return 0.1
    
    #stuck on edge of box
    elif(x == x_next and x == 29):
        return 0.3
    elif(x==x_next and x==0):
        return 0.2
    elif(y==y_next and y==0):
        return 0.1
    elif(y==y_next and y==29):
        return 0.4
    else:
        return 0


def HMM_filter(Obs_E, Bel):
<<<<<<< HEAD
    Bel_t_1 = np.zeros((30,30), dtype = np.float64)
=======
    Bel_t_1 = np.zeros((30,30))
>>>>>>> rocktim
    for i in range(30):
        for j in range(30):
            if(j-1 >= 0):
                Bel_t_1[i,j] += Bel[i, j-1] * Motion_model((i, j-1), (i,j))
            if(j + 1 < 30):
                Bel_t_1[i,j] += Bel[i, j+1] * Motion_model((i, j+1), (i,j))
            if(i-1 >= 0):
                Bel_t_1[i,j] += Bel[i-1, j] * Motion_model((i-1,j), (i,j))
            if(i+1 < 30):
                Bel_t_1[i,j] += Bel[i+1, j] * Motion_model((i+1, j), (i, j))
    eta = 0
    for i in range(30):
        for j in range(30):
            ProbE_X = Sensor_model(Obs_E, (i,j))
            Bel[i,j] = Bel_t_1[i,j]*ProbE_X
            eta += Bel[i,j]
    Bel = Bel/eta
    return Bel


N = 25
curr_pos = None
POSITIONS = []
READINGS = []
for T in range(N):
    if (curr_pos is None):
        curr_pos = (np.random.randint(lim[0], lim[1] + 1), np.random.randint(lim[2], lim[3] + 1))
    else:
        curr_pos = move(curr_pos)

    POSITIONS.append(curr_pos)

    readings = sense(curr_pos, sensors)
    READINGS += [readings]





M = 25
Estimated_Position = []
for T in range(N):
    Bel = HMM_filter(READINGS[T], Bel)
    print(Bel)
print("Estimated Positions")
print(Estimated_Position)
