import numpy as np, matplotlib.pyplot as plt 
from scipy.constants import g

'''Define Variables'''
# Define Angular Velocity (rad/sec)
w1 = 1.5
w2 = 2

# Define Mass (kg)
m1 = 1
m2 = 2

# Define Angles of the Double Pendulum (rad)
theta1 = np.radians(100)
theta2 = np.radians(90)

# Length of the Rods (meters)
L1 = 1 
L2 = 1 

'''Define Four First-Order Differential Equations'''
theta_dot_1 = w1 # First First-Order Differential Equation (theta_dot_1)
theta_dot_2 = w2 # Second First-Order Differential Equation (theta_dot_2)

def w1_point(theta1, theta2, w1, w2): # Third First-Order Differential Equation (w1_dot)
    return (-g*(2*m1+m2)*np.sin(theta1)-m2*g*np.sin(theta1 - 2*theta2) - 
            2*np.sin(theta1-theta2)*m2*((w2**2)*L2 + 
            (w1**2)*L1*np.cos(theta1-theta2)))/(L1*(2*m1+m2-m2*np.cos(2*(theta1-theta2))))

def w2_point(theta1, theta2, w1, w2): # Fourth First-Order Differential Equation (w2_dot)
    return (2*np.sin(theta1-theta2)*((w1**2)*L1*(m1+m2)+g*(m1+m2)*np.cos(theta1)+
            ((w2**2)*L2*m2*np.cos(theta1-theta2))))/(L2*(2*m1+m2-m2*np.cos(2*(theta1-theta2))))

'''Combine Four First-Order Differential Equations into One Equation'''
state = np.array([theta1, theta2, w1, w2]) # Store the angles and angular velocities in an array
def double_pendulum_derivative(t, state): # Define a function that calculates the four first-order differential equations
    theta1, theta2, w1, w2 = state 
    theta_dot_1 = w1 
    theta_dot_2 = w2 
    w1_dot = w1_point(theta1, theta2, w1, w2)
    w2_dot = w2_point(theta1, theta2, w1, w2)
    return np.array([theta_dot_1, theta_dot_2, w1_dot, w2_dot]) 

'''Define Time Steps'''
ti, tf = 0, 15 # Initial and Final Time Value
number_of_steps = 400 # Number of Time Steps
h = (tf-ti)/number_of_steps # Delta T
time_values = np.linspace(ti, tf, number_of_steps + 1) # Define an array of time values

'''Runge-Kutta 4th Order'''
def runge_kutta_4th_order(f, time_values, state):
    rk4_state = state.copy()
    rk4_theta1_values = np.zeros(len(time_values))
    rk4_theta2_values = np.zeros(len(time_values))
    rk4_w1_values = np.zeros(len(time_values))
    rk4_w2_values = np.zeros(len(time_values))
    for i, t in enumerate(time_values):
        k1 = f(t, rk4_state)
        k2 = f(t + 0.5 * h, rk4_state + (h / 2) * k1)
        k3 = f(t + 0.5 * h, rk4_state + (h / 2) * k2)
        k4 = f(t+h, rk4_state + h*k3)
        rk4_state = rk4_state + ((k1 + 2 * k2 + 2 * k3 + k4) * (h / 6))
        rk4_theta1_values[i] = rk4_state[0]
        rk4_theta2_values[i] = rk4_state[1]
        rk4_w1_values[i] = rk4_state[2]
        rk4_w2_values[i] = rk4_state[3]
    return rk4_theta1_values, rk4_theta2_values, rk4_w1_values, rk4_w2_values

'''Compute Runge Kutta 4th Order Step'''
# Compute Runge Kutta 4th Order Step (Initial State)
rk4_theta1_values, rk4_theta2_values, rk4_w1_values, \
    rk4_w2_values = runge_kutta_4th_order(double_pendulum_derivative, time_values, state)

# Compute Runge Kutta 4th Order Step (Warmed-up State)
warmed_state = state + state*1e-8
warmed_rk4_theta1_values, warmed_rk4_theta2_values, warmed_rk4_w1_values, \
    warmed_rk4_w2_values = runge_kutta_4th_order(double_pendulum_derivative, time_values, warmed_state)

'''Compute Position of Masses using RK4 Method (Polar to Rectangular)'''
def mass_position(theta1_value, theta2_value): 
    x1 = L1 * np.sin(theta1_value)
    y1 = -1*L1 * np.cos(theta1_value)
    x2 = x1 + L2 * np.sin(theta2_value)
    y2 = y1 + -1*L2 * np.cos(theta2_value)
    return x1, y1, x2, y2

# Compute Position of Masses using RK4 Method (Initial State)
rk4_x1, rk4_y1, rk4_x2, rk4_y2 = mass_position(rk4_theta1_values, rk4_theta2_values)

# Compute Position of Masses using RK4 Method (Warmed-up State)
warmed_rk4_x1, warmed_rk4_y1, warmed_rk4_x2, warmed_rk4_y2 = mass_position(warmed_rk4_theta1_values, warmed_rk4_theta2_values)

'''Compute Difference Trajectory'''
# Compute Difference Trajectory for Mass 1
difference_rk4_x1 = np.abs(warmed_rk4_x1 - rk4_x1)
difference_rk4_y1 = np.abs(warmed_rk4_y1 - rk4_y1)
difference_1 = np.sqrt((difference_rk4_x1**2) + (difference_rk4_y1**2))
normalized_difference_1 = difference_1 / np.linalg.norm(difference_1)

# Plot the Difference Trajectory over Time for Mass 1
plt.semilogy(normalized_difference_1)
plt.xlabel('Number of Time Steps')
plt.ylabel('Normalized Difference for Mass 1')
plt.show() 

# Compute Difference Trajectory for Mass 2
difference_rk4_x2 = np.abs(warmed_rk4_x2 - rk4_x2)
difference_rk4_y2 = np.abs(warmed_rk4_y2 - rk4_y2)
difference_2 = np.sqrt((difference_rk4_x2**2) + (difference_rk4_y2**2))
normalized_difference_2= difference_2 / np.linalg.norm(difference_2)

# Plot the Difference Trajectory over Time for Mass 2
plt.semilogy(normalized_difference_2)
plt.xlabel('Number of Time Steps')
plt.ylabel('Normalized Difference for Mass 2')
plt.show() 

'''Compute the Lyapunov Exponent'''
# Compute Lyapunov Exponent for Mass 1 
lyapunov_exponent_1, y_int_1 = np.polyfit(time_values, np.log(normalized_difference_1), 1)
print(f'Lyapunov Exponent for Mass 1: {lyapunov_exponent_1}')

# Compute Lyapunov Exponent for Mass 2
lyapunov_exponent_2, y_int_2 = np.polyfit(time_values, np.log(normalized_difference_2), 1)
print(f'Lyapunov Exponent for Mass 2: {lyapunov_exponent_2}')

# Plot Graphs as Subplots
fig, axis = plt.subplots(1, 2)
axis[0].semilogy(normalized_difference_1)
axis[0].set_xlabel('Number of Time Steps')
axis[0].set_ylabel('Normalized Difference for m₁')
axis[0].set_title("Difference Trajectory over Time for m₁", fontsize = 10)
axis[1].semilogy(normalized_difference_2)
axis[1].set_xlabel('Number of Time Steps')
axis[1].set_ylabel('Normalized Difference for m₂')
axis[1].set_title("Difference Trajectory over Time for m₂", fontsize = 10)
plt.show()


