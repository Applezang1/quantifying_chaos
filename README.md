# Quantitatively and Visually Analyzing Chaos in the Double Pendulum 

This repository is a collection of my code, documentation, and results for the 2026 Los Angeles County Science & Engineering Fair (LACSEF). 

## Abstract 

Chaos, discovered in the 1960s by Edward N. Lorenz, has a broad variety of applications from describing many systems in our daily life to analyzing systems such as electrical pulses in the heart and concepts in quantum mechanics. Chaos describes a minuscule change from initial conditions of a system that results in a largely different outcome. A well-known example of chaos theory is the butterfly effect, where a butterfly that flaps its wings leads to the formation of a tornado. However, chaotic systems can be difficult to perceive. This project aims to quantify the chaos of a double pendulum and make it easier to grasp to non-experts through visual models. Three different numerical integrators were tested to simulate a double pendulum: first-order Euler method, second-order Runge-Kutta (RK2) and fourth-order Runge-Kutta (RK4). Based on our data, RK4 was the most accurate and was chosen to simulate the double pendulum. The divergence plots, Poincaré maps, chaos maps, and Lyapunov exponents clearly show the chaotic nature of the system. Specifically, the Lyapunov exponent quantifies a system’s chaos, and represents this as two distinct values, 1.314 for mass 1 and 1.267 for mass 2. The positive Lyapunov exponent defines the system as chaotic. Additionally, the chaos map visually shows the overall chaos and unpredictability of a double pendulum in an understandable way. This project was successful in meeting our objective by clearly displaying the chaos of a double pendulum and showing that chaos can be deterministic, quantifiable and most importantly, understandable.

## Repository Format
### Overview and Summary of Sections 

The main folders of this repository are listed below 

* [**Assets**](./Assets) - Contains the images referenced in the documentation, the final board picture, and the procedure flowchart.

* [**Code**](./Code) - Contains the code used for simulating the double pendulum, computing the Lyapunov Exponent, and plotting the Poincare sections. 

* [**Derivation**](./Derivation) - Contains the derivation of the Euler-Lagrange Equation and the Lyapunov Exponent.

* [**Documentation**](./Documentation) - Contains the sections of the final report. The sections are listed below.

    * [**Abstract**](./Documentation/abstract.md)

    * [**Research**](./Documentation/research.md) 

    * [**Materials**](./Documentation/materials.md) 

    * [**Procedure**](./Documentation/procedure.md)

    * [**Results**](./Documentation/results.md)

    * [**Conclusion**](./Documentation/conclusion.md) 

    * [**Works Cited**](./Documentation/works_cited.md)    

    