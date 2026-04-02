import numpy as np, matplotlib.pyplot as plt 
from scipy.constants import g

'''Compute the Poincare Section for Chaotic Initial Conditions'''
### Define Chaotic Initial Condition Variables ###
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

### Define Four First-Order Differential Equations ###
theta_dot_1 = w1 # First First-Order Differential Equation (theta_dot_1)
theta_dot_2 = w2 # Second First-Order Differential Equation (theta_dot_2)

def w1_point(theta1, theta2, w1, w2): # Third First-Order Differential Equation (w1_dot)
    return (-g*(2*m1+m2)*np.sin(theta1)-m2*g*np.sin(theta1 - 2*theta2) - 
            2*np.sin(theta1-theta2)*m2*((w2**2)*L2 + 
            (w1**2)*L1*np.cos(theta1-theta2)))/(L1*(2*m1+m2-m2*np.cos(2*(theta1-theta2))))

def w2_point(theta1, theta2, w1, w2): # Fourth First-Order Differential Equation (w2_dot)
    return (2*np.sin(theta1-theta2)*((w1**2)*L1*(m1+m2)+g*(m1+m2)*np.cos(theta1)+
            ((w2**2)*L2*m2*np.cos(theta1-theta2))))/(L2*(2*m1+m2-m2*np.cos(2*(theta1-theta2))))

### Combine Four First-Order Differential Equations into One Equation ###
state = np.array([theta1, theta2, w1, w2]) # Store the angles and angular velocities in an array
def double_pendulum_derivative(t, state): # Define a function that calculates the four first-order differential equations
    theta1, theta2, w1, w2 = state 
    theta_dot_1 = w1 
    theta_dot_2 = w2 
    w1_dot = w1_point(theta1, theta2, w1, w2)
    w2_dot = w2_point(theta1, theta2, w1, w2)
    return np.array([theta_dot_1, theta_dot_2, w1_dot, w2_dot]) 

### Define Time Steps ###
ti, tf = 0, 3000 # Initial and Final Time Value
number_of_steps = 3*100000 # Number of Time Steps
h = (tf-ti)/number_of_steps # Delta T
time_values = np.linspace(ti, tf, number_of_steps + 1) # Define an array of time values

### Runge-Kutta 4th Order ###
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

# Compute Runge Kutta 4th Order Step (Chaotic Initial Conditions)
rk4_theta1_values, rk4_theta2_values, rk4_w1_values, \
    rk4_w2_values = runge_kutta_4th_order(double_pendulum_derivative, time_values, state)

# Compute Hyperplane Intersection Data Points for Mass 1 (Chaotic Initial Conditions)
intersections = np.where(np.diff(np.sign(rk4_theta2_values)))[0]
first_intersections = np.where(np.diff(np.sign(rk4_theta2_values-2*np.pi)))[0]
second_intersections = np.where(np.diff(np.sign(rk4_theta2_values-4*np.pi)))[0] 
third_intersections = np.where(np.diff(np.sign(rk4_theta2_values-6*np.pi)))[0] 
fourth_intersections = np.where(np.diff(np.sign(rk4_theta2_values-8*np.pi)))[0] 
fifth_intersections = np.where(np.diff(np.sign(rk4_theta2_values+2*np.pi)))[0]
sixth_intersections = np.where(np.diff(np.sign(rk4_theta2_values+4*np.pi)))[0] 
seventh_intersections = np.where(np.diff(np.sign(rk4_theta2_values+6*np.pi)))[0] 
eight_intersections = np.where(np.diff(np.sign(rk4_theta2_values+8*np.pi)))[0] 
nine_intersections = np.where(np.diff(np.sign(rk4_theta2_values-10*np.pi)))[0] 
ten_intersections = np.where(np.diff(np.sign(rk4_theta2_values-12*np.pi)))[0] 
eleven_intersections = np.where(np.diff(np.sign(rk4_theta2_values-14*np.pi)))[0] 

# Combine all intersection indices (0, 2π, 4π, 6π, and 8π crossings)
intersections = np.concatenate(
    (
        intersections,
        first_intersections,
        second_intersections,
        third_intersections,
        fourth_intersections,
        fifth_intersections, 
        sixth_intersections, 
        seventh_intersections, 
        eight_intersections, 
        nine_intersections, 
        ten_intersections, 
        eleven_intersections
    )
)
intersections = np.unique(intersections)
intersections = intersections[rk4_w2_values[intersections] > 0]
mass1_intersected_theta1 = rk4_theta1_values[intersections]
mass1_intersected_w1 = rk4_w1_values[intersections]

# Compute Hyperplane Intersection Data Points for Mass 1
mass1_intersected_theta1 = (mass1_intersected_theta1 + 2*np.pi) % (4*np.pi) - 2*np.pi

'''Compute the Poincare Section for Non-Chaotic Initial Conditions'''
### Define Non-Chaotic Initial Condition Variables ###
# Define Angular Velocity (rad/sec)
w1 = 1.5
w2 = 2

# Define Mass (kg)
m1 = 1
m2 = 2

# Define Angles of the Double Pendulum (rad)
theta1 = np.radians(10)
theta2 = np.radians(5)

# Length of the Rods (meters)
L1 = 1 
L2 = 1 

### Combine Four First-Order Differential Equations into One Equation ###
state = np.array([theta1, theta2, w1, w2]) # Store the angles and angular velocities in an array

### Define Time Steps ###
ti, tf = 0, 1000 # Initial and Final Time Value
number_of_steps = 300000 # Number of Time Steps
h = (tf-ti)/number_of_steps # Delta T
time_values = np.linspace(ti, tf, number_of_steps + 1) # Define an array of time values

# Compute Runge Kutta 4th Order Step (Non-Chaotic Initial Conditions)
rk4_theta1_values, rk4_theta2_values, rk4_w1_values, \
    rk4_w2_values = runge_kutta_4th_order(double_pendulum_derivative, time_values, state)

# Compute Hyperplane Intersection Data Points for Mass 1 (Non-Chaotic Initial Conditions)
intersections = np.where(np.diff(np.sign(rk4_theta2_values)))[0]
intersections = intersections[rk4_w2_values[intersections] > 0]
mass1_intersected_w1_2 = rk4_w1_values[intersections]
mass1_intersected_theta1_2 = rk4_theta1_values[intersections]
mass1_intersected_theta1_2 = (mass1_intersected_theta1_2 + 2*np.pi) % (4*np.pi) - 2*np.pi

'''Plot Poincare Section of Non-Chaotic and Chaotic Initial Conditions'''
fig, axis = plt.subplots(1, 2)
axis[0].scatter(mass1_intersected_theta1, mass1_intersected_w1)
axis[0].set_xlabel("θ₁")
axis[0].set_ylabel("ω₁") 
axis[0].set_title("Poincare Section of Chaotic Initial Conditions", fontsize = 10)
axis[1].scatter(mass1_intersected_theta1_2, mass1_intersected_w1_2)
axis[1].set_xlabel("θ₁")
axis[1].set_ylabel("ω₁") 
axis[1].set_title("Poincare Section of Non-Chaotic Initial Conditions", fontsize = 10)
plt.show()
