import numpy as np
import matplotlib.pyplot as plt
from plot_uncertainty_ellipse import confidence_ellipse
from scipy.stats import multivariate_normal as normal 
stdev = 10.0

def simulate(init_position = np.array([10, 10]), init_velocity= np.random.rand(2) * 5, control_input = np.array([0, 0]), delta_time= 1.0,
             Rt =  np.diag([1.0, 1.0, 0.01, 0.01]), Qt = np.eye(2) * (stdev**2), T_step = 200):
    initial_position = init_position
    u = control_input
    initial_velocity = init_velocity  # random velocity, [v_x, v_y]

    initial_state = np.stack([initial_position, initial_velocity]).reshape(-1)
    T = T_step  # number of time steps
    delt = delta_time

    R = Rt
    delta = Qt

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
        current_state = current_state @ A.T + u @ B.T + np.random.multivariate_normal([0, 0, 0, 0], R, 1).reshape(-1)  # current position update.
        measurement_update = current_state @ C.T + np.random.multivariate_normal([0, 0], delta, 1).reshape(-1)

        positions += [[current_state[0], current_state[1]]]
        measurements += [[measurement_update[0], measurement_update[1]]]

    print(positions)
    print(measurements)

    positions = np.array(positions)
    measurements = np.array(measurements)

    return positions, measurements

#positions, measurements = simulate(init_position = np.array([10, 10]), init_velocity= np.random.rand(2) * 5, control_input = np.array([0, 0]), delta_time= 1,
#             Rt =  np.diag([1.0, 1.0, 0.01, 0.01]), Qt = np.eye(2) * 100, T_step = 200)


initial_position_1 = np.array([10, 10])
u_1 = np.array([0, 0])
initial_velocity_1 = np.random.rand(2) * 5  # random velocity, [v_x, v_y]
initial_state_1 = np.stack([initial_position_1, initial_velocity_1]).reshape(-1)
T = 200

positions_1, measurements_1 = simulate(init_position = initial_position_1, init_velocity=initial_velocity_1, control_input = np.array([0,0]), delta_time = 1, Rt =  np.diag([1.0, 1.0, 0.01, 0.01]), Qt = np.eye(2) * (stdev**2), T_step = 200)

initial_position_2 = np.array([50,50])
u_2 = np.array([0,0])
initial_velocity_2 = np.random.rand(2)*5 
initial_state_2 = np.stack([initial_position_2, initial_velocity_2]).reshape(-1)

positions_2, measurements_2 = simulate(init_position = initial_position_2, init_velocity=initial_velocity_2, control_input = np.array([0,0]), delta_time = 1, Rt =  np.diag([1.0, 1.0, 0.01, 0.01]), Qt = np.eye(2) * (stdev**2), T_step = 200)

def Kalman_filter(mu_1_t_1, sig_1_t_1, mu_2_t_1, sig_2_t_1, u_t, z_t_a, z_t_b, del_t = 1.0, nearest=True):
    A_t = np.array([[1,0,del_t,0],[0,1,0,del_t],[0 ,0,1,0],[0,0,0,1]], dtype = np.float64)
    B_t = np.array([[0,0],[0,0],[1,0],[0,1]], dtype = np.float64)
    C_t = np.array([[1,0,0,0],[0,1,0,0]], dtype = np.float64)
    R_t = np.array(np.diag([1.0, 1.0, 0.01, 0.01]), dtype = np.float64)
    
    mu_1_t_dash = np.matmul(A_t,mu_1_t_1) + np.matmul(B_t, u_t)
    sig_1_t_dash = np.matmul(A_t, np.matmul(sig_1_t_1, A_t.T)) + R_t
    
    mu_2_t_dash = np.matmul(A_t,mu_2_t_1) + np.matmul(B_t, u_t)
    sig_2_t_dash = np.matmul(A_t, np.matmul(sig_2_t_1, A_t.T)) + R_t    
    
    Q_t = np.eye(2)*(stdev**2)

    #data association step!
    if(nearest):
        if(np.sum((z_t_a - mu_1_t_dash[:2])**2) < np.sum((z_t_b - mu_1_t_dash[:2])**2)):
            z_1_t = z_t_a
            z_2_t = z_t_b
        else:
            z_1_t = z_t_b
            z_2_t = z_t_a

    else:
        #case 1 : a -> 1 and b -> 2
        p1 = normal.pdf(z_t_a, mean=np.matmul(C_t, mu_1_t_dash), cov=np.matmul(C_t, np.matmul(sig_1_t_dash, C_t.T)))*normal.pdf(z_t_b, mean=np.matmul(C_t, mu_2_t_dash), cov=np.matmul(C_t, np.matmul(sig_2_t_dash, C_t.T)))
        #case 2: a-> 2, b-> 1
        p2 = normal.pdf(z_t_b, mean=np.matmul(C_t, mu_1_t_dash), cov=np.matmul(C_t, np.matmul(sig_1_t_dash, C_t.T)))*normal.pdf(z_t_a, mean=np.matmul(C_t, mu_2_t_dash), cov=np.matmul(C_t, np.matmul(sig_2_t_dash, C_t.T)))

        if(p1 > p2):
            z_1_t = z_t_a
            z_2_t = z_t_b
        else:
            z_1_t = z_t_b
            z_2_t = z_t_a
    interm_1 = np.linalg.inv(np.matmul(C_t, np.matmul(sig_1_t_dash, C_t.T)) + Q_t)
    K_1_t = np.matmul(sig_1_t_dash, np.matmul(C_t.T,interm_1))
    mu_1_t = mu_1_t_dash + np.matmul(K_1_t, (z_1_t - np.matmul(C_t, mu_1_t_dash)))
    sig_1_t = sig_1_t_dash - np.matmul(K_1_t, np.matmul(C_t,sig_1_t_dash))

    interm_2 = np.linalg.inv(np.matmul(C_t, np.matmul(sig_2_t_dash, C_t.T)) + Q_t)
    K_2_t = np.matmul(sig_2_t_dash, np.matmul(C_t.T,interm_2))
    mu_2_t = mu_2_t_dash + np.matmul(K_2_t, (z_2_t - np.matmul(C_t, mu_2_t_dash)))
    sig_2_t = sig_2_t_dash - np.matmul(K_2_t, np.matmul(C_t,sig_2_t_dash))

    return  mu_1_t,sig_1_t, mu_2_t, sig_2_t

mu_1_0 = initial_state_1
sig_1_0 = np.eye(4)*0.0001
estimated_positions_1 = [[mu_1_0[0], mu_1_0[1]]]

mu_1_t = initial_state_1
sig_1_t = np.eye(4)

mu_2_0 = initial_state_2
sig_2_0 = np.eye(4)*0.0001
estimated_positions_2 = [[mu_2_0[0], mu_2_0[1]]]
mu_2_t = initial_state_2
sig_2_t = np.eye(4)

for t in range(1, T + 1):
    p = np.random.rand(1)
    if(p > 0.5):
        z_t_a = measurements_1[t].T
        z_t_b = measurements_2[t].T
    else:
        z_t_a = measurements_2[t].T
        z_t_b = measurements_1[t].T
    
    u_t = np.array([0,0]).T
    if t == 0:
        mu_1_t_1 = mu_1_0
        sig_1_t_1 = sig_1_0
        mu_2_t_1 = mu_2_0
        sig_2_t_1 = sig_2_0
    else:
        mu_1_t_1 = mu_1_t
        sig_1_t_1 = sig_1_t
        mu_2_t_1 = mu_2_t
        sig_2_t_1 = sig_2_t

    mu_1_t, sig_1_t, mu_2_t, sig_2_t = Kalman_filter(mu_1_t_1,sig_1_t_1, mu_2_t_1, sig_2_t_1, u_t,z_t_a, z_t_b,1.0, nearest=False)

    estimated_positions_1 += [[mu_1_t[0], mu_1_t[1]]]
    estimated_positions_2 += [[mu_2_t[0], mu_2_t[1]]]

print(positions_1)
print(positions_2)
estimated_positions_1 = np.array(estimated_positions_1)
estimated_positions_2 = np.array(estimated_positions_2)
print(estimated_positions_1, estimated_positions_2)

fig, ax = plt.subplots()
ax.plot(positions_1[:, 0], positions_1[:, 1], 'r')
ax.plot(positions_2[:, 0], positions_2[:, 1], 'g')
#plt.show()
ax.plot(estimated_positions_1[:, 0], estimated_positions_1[:, 1], 'b')
ax.plot(estimated_positions_2[:, 0], estimated_positions_2[:, 1], 'black')
plt.show()

plt.clf(); plt.cla()
