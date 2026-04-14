# Procedure/Methods 

1. Initialize chaotic initial conditions for the angular velocity (rad/sec), mass (kg), angle (radians), and the length of rod (meters) for each mass in the double pendulum setup.

2. Define the four first-order differential equations that describe the double pendulum’s motion in its respective variables.

3. Define an array of time steps, where each time step is separated from each other by a value (Δt). The position and velocity of the masses will be computed at each time step.

4. Implement the Euler Method, Runge-Kutta 2nd Order, and the Runge-Kutta 4th Order as a set of numerical integrators to simulate the double pendulum’s motion by using its differential equations.

5. Compute the angle and angular velocities of each mass at each time point for all the numerical integrators in the defined set.

6. Using the polar to rectangular coordinate formula, convert the θ₁ and θ₂ values into rectangular coordinates for mass 1 and mass 2 for each numerical integrator 

7. Compute the velocity using the rectangular coordinates for mass 1 and mass 2 and use the velocity to compute the potential energy and kinetic energy for each numerical integrator. Summing the potential and kinetic energy returns the total energy of the system.

8. Plot θ₁ vs θ₂ for each numerical integrator using matplotlib’s plotting functions.  

9. With the total energy of the system for each numerical integrator, plot the total energy of system vs time using matplotlib’s plotting function.  

10. Identify the numerical integrator that demonstrates the most stable behavior using the total energy plot created in step 9. 

11. Using the chosen numerical integrator, compute θ₁ and θ₂ and convert it to rectangular coordinates (as done above). 

12. Repeat step 11, but offset the set of parameters of the double pendulum by a negligible difference (1e-8). Convert the angle values into rectangular coordinates.

13. Compute the Euclidean distance between the computed rectangular coordinates.

14. Plot the log of the difference trajectory over time using matplotlib’s plotting function.

15. Fit a linear function on the plot defined in step 16. The slope of the linear function is defined as the Lyapunov exponent. 

16. Compute the θ₁ and ω₁ when θ₂ = 0 and ω₂ is positive.

17. Plot a Poincaré graph with the computed θ₁ and ω₁ values as the respective x and y coordinates.

18. Repeat steps 16-17 for non-chaotic initial conditions.
