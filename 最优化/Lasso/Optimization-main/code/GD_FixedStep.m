function f = GD_FixedStep(Nit, A, x, step)
f = zeros(Nit, 1);
for i_it = 1:Nit
    f(i_it) = Quadratic(A, x);
    nabla = A*x;
    x = x - step*nabla;  
end
end