# Gradient Descent and ADMM for LASSO
A simple example of Gradient Descent algorithm and ADMM algorithm.(Homework of the large-scale optimization course.)
Take the Gradient Descent algorithm as an example.
I compared different stepsize selection methods, which are fixed stepsize, Armijo-Goldstein rule and Wolfe-Powell rule, in this simple Gradient Descent algorithm.
My goal is to minimize the quadratic function $$f(\mathbf{x})=\frac{1}{2}\mathbf{x}^H\mathbf{A}\mathbf{x}$$, where $\mathbf{A}$ is randomly generated and positive semidefinite.

Some simulation results are placed here, all the code can be found in the 'code' folder

<div align=center>
<img src="https://github.com/LiZhuoRan0/SimpleGradientDescent/blob/main/SomeOfSimulations/N_10.jpg"/>
</div>

<div align=center>
<img src="https://github.com/LiZhuoRan0/SimpleGradientDescent/blob/main/SomeOfSimulations/N_100.jpg"/>
</div>
