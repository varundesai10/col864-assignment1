import numpy as np

lim = [0,29,0,29]
move_probabilies = [0.4, 0.1, 0.2, 0.3] #up, down, left, right
move_dict = {0:lambda x, y: (x, min(y+1, lim[3])) , 1:lambda x, y: (x, max(y-1, lim[2])), 2:lambda x, y: (max(x-1, lim[0]),y), 3:lambda x, y: (min(x+1, lim[1]), y)}

sensors = [(8, 15),(15, 15),(22, 15),(15, 22)]
p_sensor = [0.9, 0.8, 0.7, 0.6, 0.5]

curr_pos = (0,0)

def move(curr_pos):
    x, y = curr_pos
    f = move_dict[np.random.choice([0,1,2,3], p=move_probabilies)]
    return f(x, y)

def sense(curr_pos, sensors):
    x, y = curr_pos
    sensor_readings = []
    for sens in sensors:
        x_, y_ = sens
        a = int(max(np.abs(x-x_), np.abs(np.abs(y-y_))))
        if a < len(p_sensor):
            p = p_sensor[a]
        else:
            p = 0
        reading = np.random.choice([0,1], p=[1-p, p])
        sensor_readings.append(reading)
    return sensor_readings


N = 25
curr_pos = None
POSITIONS = []
READINGS = []
for T in range(N):
    if(curr_pos is None):
        curr_pos = (np.random.randint(lim[0], lim[1] + 1), np.random.randint(lim[2], lim[3] + 1))
    else:
        curr_pos = move(curr_pos)
    
    POSITIONS.append(curr_pos)
    print('moved to {}'.format(curr_pos))
    
    readings = sense(curr_pos, sensors)
    READINGS += [readings]
    print('readings = {}'.format(readings))

print('Final Print')
print('Positions = {} \nReadings = {}'.format(POSITIONS, READINGS))
    