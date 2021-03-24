import numpy as np
import matplotlib.pyplot as plt
import math as mt

wavelength = 5.0
var = 1
def simulate(init_position = np.array([10, 10]), init_velocity= np.random.rand(2) * 5, delta_time= 1.0, control=0,
             Rt =  np.diag([1.0, 1.0, 0.01, 0.01]), Qt = np.eye(2) * var, T_step = 200):

    initial_position = init_position

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
        time = t * delta_time
        if control == 0:
            u = np.array([0 ,0])
        if control == 1:
            u = np.array([np.sin(time/wavelength), np.cos(time/wavelength)])
        current_state = current_state @ A.T + u @ B.T + np.random.multivariate_normal([0, 0, 0, 0], R, 1).reshape(
            -1)  # current position update.
        measurement_update = current_state @ C.T + np.random.multivariate_normal([0, 0], delta, 1).reshape(-1)

        positions += [[current_state[0], current_state[1]]]
        measurements += [[measurement_update[0], measurement_update[1]]]

    print(positions)
    print(measurements)

    positions = np.array(positions)
    measurements = np.array(measurements)

    return positions, measurements

def Kalman_filter(mu_t_1, sig_t_1, u_t, z_t, del_t = 1.0):
    
    A_t = np.array([[1,0,del_t,0],[0,1,0,del_t],[0 ,0,1,0],[0,0,0,1]], dtype = np.float64)
    B_t = np.array([[0,0],[0,0],[1,0],[0,1]], dtype = np.float64)
    C_t = np.array([[1,0,0,0],[0,1,0,0]], dtype = np.float64)
    Q_t = np.diag([1.0, 1.0, 0.01, 0.01])

    mu_t_dash = np.matmul(A_t,mu_t_1) + np.matmul(B_t, u_t)
    sig_t_dash = np.matmul(A_t, np.matmul(sig_t_1, A_t.T)) + Q_t
    
    R_t = np.eye(2)*var
    interm = np.linalg.inv(np.matmul(C_t, np.matmul(sig_t_dash, C_t.T)) + R_t)
    K_t = np.matmul(sig_t_dash, np.matmul(C_t.T,interm))
    mu_t = mu_t_dash + np.matmul(K_t, (z_t - np.matmul(C_t, mu_t_dash)))
    sig_t = sig_t_dash - np.matmul(K_t, np.matmul(C_t,sig_t_dash))
    
    return  mu_t,sig_t



if __name__ == '__main__':
    delta_time= 1.0
    initial_position = np.array([10, 10])
    u = np.array([0, 0])
    initial_velocity = np.random.rand(2) * 5  # random velocity, [v_x, v_y]
    initial_state = np.stack([initial_position, initial_velocity]).reshape(-1)
    T = 200
    positions, measurements = simulate( init_position = initial_position, 
                                        init_velocity= initial_velocity, 
                                        delta_time= delta_time, control= 1, 
                                        Rt =  np.diag([1.0, 1.0, 0.01, 0.01]), 
                                        Qt = np.eye(2) * var, 
                                        T_step = 200)

    mu_0 = initial_state
    sig_0 = np.eye(4)*0.0001
    estimated_positions = [[mu_0[0], mu_0[1]]]
    mu_t = initial_state
    sig_t = np.eye(4)
    for t in range(T):

        z_t = measurements[t]
        time = t * delta_time
        u_t = np.array([np.sin(time/wavelength), np.cos(time/wavelength)]).T
        if t == 0:
            mu_t_1 = mu_0
            sig_t_1 = sig_0
        else:
            mu_t_1 = mu_t
            sig_t_1 = sig_t
        mu_t, sig_t = Kalman_filter(mu_t_1,sig_t_1,u_t,z_t,delta_time)
        estimated_positions += [[mu_t[0], mu_t[1]]]
    print(positions)
    estimated_positions = np.array(estimated_positions)
    print(estimated_positions)

    plt.plot(measurements[:, 0], measurements[:, 1], 'g')
    #plt.show()
    plt.plot(positions[:, 0], positions[:, 1], 'r')
    #plt.show()
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'b')
    plt.show()

    euclidian = np.sqrt(np.sum((positions - estimated_positions)**2, axis=1))
    plt.plot(euclidian); plt.title(f'Average euclidian distance is {np.mean(euclidian)}')
    plt.show()
    print(euclidian)
