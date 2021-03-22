import numpy as np
import matplotlib.pyplot as plt

initial_position = np.array([10, 10])
u = np.array([0, 0])
initial_velocity = np.random.rand(2) * 5  # random velocity, [v_x, v_y]

initial_state = np.stack([initial_position, initial_velocity]).reshape(-1)
sigma = (10) ** 2
T = 200  # number of time steps
delt = 1.0

R = np.diag([1.0, 1.0, 0.01, 0.01])
delta = np.eye(2) * sigma

positions = [[initial_position[0], initial_position[1]]]
initial_measurement = initial_position + np.random.multivariate_normal([0, 0], delta, 1).reshape(-1)
measurements = [[initial_measurement[0], initial_measurement[1]]]
# motion model, x_t+1 = At * xt + Bt * ut + eps
# sensor model, z_t = C_t * x_t + del_t

A = np.array([[1, 0, delt, 0], [0, 1, 0, delt], [0, 0, 1, 0], [0, 0, 0, 1]])
B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

current_state = initial_state
for t in range(T):
    current_state = current_state @ A.T + u @ B.T + np.random.multivariate_normal([0, 0, 0, 0], R, 1).reshape(
        -1)  # current position update.
    measurement_update = current_state @ C.T + np.random.multivariate_normal([0, 0], delta, 1).reshape(-1)

    positions += [[current_state[0], current_state[1]]]
    measurements += [[measurement_update[0], measurement_update[1]]]

print(positions)
print(measurements)

positions = np.array(positions)
measurements = np.array(measurements)
plt.plot(positions[:, 0], positions[:, 1], 'r')
plt.plot(measurements[:, 0], measurements[:, 1], 'b')