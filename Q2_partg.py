import numpy as np
import matplotlib.pyplot as plt
import math as mt
from Q2_partd import simulate

delta_time= 1
positions, measurements = simulate(init_position = np.array([10, 10]), init_velocity= np.random.rand(2) * 5, delta_time= 1, control= 1,
             Rt =  np.diag([1.0, 1.0, 0.01, 0.01]), Qt = np.eye(2) * 100, T_step = 200)


initial_position = np.array([10, 10])
u = np.array([0, 0])
initial_velocity = np.random.rand(2) * 5  # random velocity, [v_x, v_y]
initial_state = np.stack([initial_position, initial_velocity]).reshape(-1)
T = 200
def Kalman_filter_partg(mu_t_1, sig_t_1, u_t, z_t, del_t = 1.0):
    A_t = np.array([[1,0,del_t,0],[0,1,0,del_t],[0 ,0,1,0],[0,0,0,1]], dtype = np.float64)
    B_t = np.array([[0,0],[0,0],[1,0],[0,1]], dtype = np.float64)
    C_t = np.array([[1,0,0,0],[0,1,0,0]], dtype = np.float64)
    Q_t = np.diag([1.0, 1.0, 0.01, 0.01])
    mu_t_dash = np.matmul(A_t,mu_t_1) + np.matmul(B_t, u_t)
    sig_t_dash = np.matmul(A_t, np.matmul(sig_t_1, A_t.T)) + Q_t
    if z_t[0] == mt.inf and z_t[1]==mt.inf:
        return mu_t_dash, sig_t_dash
    else:
        R_t = np.eye(2) * 100
        interm = np.linalg.inv(np.matmul(C_t, np.matmul(sig_t_dash, C_t.T)) + R_t)
        K_t = np.matmul(sig_t_dash, np.matmul(C_t.T, interm))
        mu_t = mu_t_dash + np.matmul(K_t, (z_t - np.matmul(C_t, mu_t_dash)))
        sig_t = sig_t_dash - np.matmul(K_t, np.matmul(C_t,sig_t_dash))
        return  mu_t,sig_t

mu_0 = initial_state
sig_0 = np.eye(4)*0.0001
estimated_positions = [[mu_0[0], mu_0[1]]]
mu_t = initial_state
sig_t = np.eye(4)
control = 1 # control decider
for t in range(T):
    time = t * delta_time
    if time >= 10 and time < 10 + delta_time*10 or time >=30 and time <30+ delta_time*10:
        z_t = np.array([mt.inf, mt.inf])
    else:
        z_t = measurements[t]

    if control == 0:
        u_t = np.array([0, 0])
    if control == 1:
        u_t = np.array([mt.sin(time), mt.cos(time)])
    if t == 0:
        mu_t_1 = mu_0
        sig_t_1 = sig_0
    else:
        mu_t_1 = mu_t
        sig_t_1 = sig_t
    mu_t, sig_t = Kalman_filter_partg(mu_t_1,sig_t_1,u_t,z_t,0.1)
    estimated_positions += [[mu_t[0], mu_t[1]]]
print(positions)
estimated_positions = np.array(estimated_positions)
print(estimated_positions)

plt.plot(measurements[:, 0], measurements[:, 1], 'g')
plt.show()
plt.plot(positions[:, 0], positions[:, 1], 'r')
plt.show()
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'b')
plt.show()