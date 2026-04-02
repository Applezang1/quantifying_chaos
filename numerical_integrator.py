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

'''Euler Step'''
def euler_step(f, time_values, state):
    euler_state = state.copy()
    euler_theta1_values = np.zeros(len(time_values))
    euler_theta2_values = np.zeros(len(time_values))
    euler_w1_values = np.zeros(len(time_values))
    euler_w2_values = np.zeros(len(time_values))
    for i, t in enumerate(time_values): 
        k1 = f(t, euler_state) 
        euler_state = euler_state + h*k1
        euler_theta1_values[i] = euler_state[0]
        euler_theta2_values[i] = euler_state[1]
        euler_w1_values[i] = euler_state[2]
        euler_w2_values[i] = euler_state[3] 
    return euler_theta1_values, euler_theta2_values, euler_w1_values, euler_w2_values

'''Runge-Kutta 2nd Order'''
def runge_kutta_2nd_order(f, time_values, state): 
    rk2_state = state.copy()
    rk2_theta1_values = np.zeros(len(time_values))
    rk2_theta2_values = np.zeros(len(time_values))
    rk2_w1_values = np.zeros(len(time_values))
    rk2_w2_values = np.zeros(len(time_values))
    for i, t in enumerate(time_values): 
        k1 = f(t, rk2_state)
        k2 = f(t + 0.5 * h, rk2_state + (h / 2) * k1) 
        rk2_state = rk2_state + h*k2
        rk2_theta1_values[i] = rk2_state[0]
        rk2_theta2_values[i] = rk2_state[1]
        rk2_w1_values[i] = rk2_state[2]
        rk2_w2_values[i] = rk2_state[3]
    return rk2_theta1_values, rk2_theta2_values, rk2_w1_values, rk2_w2_values

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

'''Compute Euler Step'''
euler_theta1_values, euler_theta2_values, euler_w1_values, \
    euler_w2_values = euler_step(double_pendulum_derivative, time_values, state)

'''Compute Runge Kutta 2nd Order Step'''
rk2_theta1_values, rk2_theta2_values, rk2_w1_values, \
    rk2_w2_values = runge_kutta_2nd_order(double_pendulum_derivative, time_values, state)

'''Compute Runge Kutta 4th Order Step'''
rk4_theta1_values, rk4_theta2_values, rk4_w1_values, \
    rk4_w2_values = runge_kutta_4th_order(double_pendulum_derivative, time_values, state)

# Plot Theta_1 vs Time for All Numerical Integrators
plt.plot(time_values, euler_theta1_values, label = 'Euler Step')
plt.plot(time_values, rk2_theta1_values, label = 'RK2 Step')
plt.plot(time_values, rk4_theta1_values, label = 'RK4 Step')
plt.ylim((euler_theta1_values).min(), (euler_theta1_values).max())
plt.xlabel('Time (seconds)')
plt.ylabel('θ₁')
plt.grid(True)
plt.legend()
plt.show()

# Plot Theta_2 vs Time for All Numerical Integrators
plt.plot(time_values, euler_theta2_values, label = 'Euler Step')
plt.plot(time_values, rk2_theta2_values, label = 'RK2 Step')
plt.plot(time_values, rk4_theta2_values, label = 'RK4 Step')
plt.ylim(euler_theta2_values.min(), euler_theta2_values.max())
plt.xlabel('Time (seconds)')
plt.ylabel('θ₂')
plt.grid(True)
plt.legend()
plt.show()

# Plot Theta_1 vs Theta_2 for RK2 and RK4
plt.plot(rk2_theta1_values, rk2_theta2_values, label = 'RK2 Step', c='darkorange')
plt.plot(rk4_theta1_values, rk4_theta2_values, label = 'RK4 Step', c='green')
plt.xlabel('θ₁')
plt.ylabel('θ₂')
plt.grid(True)
plt.legend()
plt.show()

'''Position of Masses for Double Pendelum'''
def mass_position(theta1_value, theta2_value): 
    x1 = L1 * np.sin(theta1_value)
    y1 = -1*L1 * np.cos(theta1_value)
    x2 = x1 + L2 * np.sin(theta2_value)
    y2 = y1 + -1*L2 * np.cos(theta2_value)
    return x1, y1, x2, y2

# Position of Masses using Euler Method (Polar to Rectangular)
euler_x1, euler_y1, euler_x2, euler_y2 = mass_position(euler_theta1_values, euler_theta2_values)

# Position of Masses using RK2 Method (Polar to Rectangular)
rk2_x1, rk2_y1, rk2_x2, rk2_y2 = mass_position(rk2_theta1_values, rk2_theta2_values)

# Position of Masses using RK4 Method (Polar to Rectangular)
rk4_x1, rk4_y1, rk4_x2, rk4_y2 = mass_position(rk4_theta1_values, rk4_theta2_values)

'''Velocity of Masses for Double Pendelum'''
# Functions for Component Velocity of Mass 1 and Mass 2
def component_velocity(values, time_values): 
    velocity = [] 
    for i in range(len(time_values)-1): 
        v = (values[i+1] - values[i])/(time_values[i+1]-time_values[i])
        velocity.append(v)
    return np.array(velocity)

# Function for the Magnitude of the Component Velocity
def magnitude_velocity(x_component, y_component): 
    return np.sqrt(x_component**2 + y_component**2)

# Velocity for Mass 1 and Mass 2 with Euler Method
euler_x1_velocity = component_velocity(euler_x1, time_values)
euler_y1_velocity = component_velocity(euler_y1, time_values)
euler_x2_velocity = component_velocity(euler_x2, time_values)
euler_y2_velocity = component_velocity(euler_y2, time_values)

euler_velocity_1 = magnitude_velocity(euler_x1_velocity, euler_y1_velocity)
euler_velocity_2 = magnitude_velocity(euler_x2_velocity, euler_y2_velocity)

# Velocity for Mass 1 and Mass 2 with RK2 Method
rk2_x1_velocity = component_velocity(rk2_x1, time_values)
rk2_y1_velocity = component_velocity(rk2_y1, time_values)
rk2_x2_velocity = component_velocity(rk2_x2, time_values)
rk2_y2_velocity = component_velocity(rk2_y2, time_values)

rk2_velocity_1 = magnitude_velocity(rk2_x1_velocity, rk2_y1_velocity)
rk2_velocity_2 = magnitude_velocity(rk2_x2_velocity, rk2_y2_velocity)

# Velocity for Mass 1 and Mass 2 with RK4 Method
rk4_x1_velocity = component_velocity(rk4_x1, time_values)
rk4_y1_velocity = component_velocity(rk4_y1, time_values)
rk4_x2_velocity = component_velocity(rk4_x2, time_values)
rk4_y2_velocity = component_velocity(rk4_y2, time_values)

rk4_velocity_1 = magnitude_velocity(rk4_x1_velocity, rk4_y1_velocity)
rk4_velocity_2 = magnitude_velocity(rk4_x2_velocity, rk4_y2_velocity)

'''Kinetic Energy for Double Pendelum''' 
# Function to Compute the Kinetic Energy
def kinetic_energy(mass, velocity): 
    return 0.5* mass * (velocity**2)

# Function to Compute the Potential Energy
def potential_energy(mass, y_position):
    y_position = y_position[:-1]
    return mass*g*y_position

# Compute the Total Energy of the System (Euler Method)
euler_total_kinetic = kinetic_energy(m1, euler_velocity_1) + kinetic_energy(m2, euler_velocity_2)
euler_total_potential = potential_energy(m1, euler_y1) + potential_energy(m2, euler_y2)
total_euler_energy = euler_total_kinetic + euler_total_potential

# Compute the Total Energy of the System (RK4 Method)
rk2_total_kinetic = kinetic_energy(m1, rk2_velocity_1) + kinetic_energy(m2, rk2_velocity_2)
rk2_total_potential = potential_energy(m1, rk2_y1) + potential_energy(m2, rk2_y2)
total_rk2_energy = rk2_total_kinetic + rk2_total_potential

# Compute the Total Energy of the System (RK4 Method)
rk4_total_kinetic = kinetic_energy(m1, rk4_velocity_1) + kinetic_energy(m2, rk4_velocity_2)
rk4_total_potential = potential_energy(m1, rk4_y1) + potential_energy(m2, rk4_y2)
total_rk4_energy = rk4_total_kinetic + rk4_total_potential

# Plot Change in Total Energy of System over Time (All Methods)
plt.plot(time_values[:-1], total_euler_energy, label = 'Euler Step')
plt.plot(time_values[:-1], total_rk2_energy, label = 'RK2 Step')
plt.plot(time_values[:-1], total_rk4_energy, label = 'RK4 Step')
margin = 0.20 * (total_euler_energy.max() - total_euler_energy.min())
plt.ylim(total_euler_energy.min() - margin, total_euler_energy.max())
plt.xlabel('Time (seconds)')
plt.ylabel('Total Energy')
plt.grid(True)
plt.legend()
plt.show()








