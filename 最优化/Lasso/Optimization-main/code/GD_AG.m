function f = GD_AG(Nit, rho, alpha_ini, A, x)
f = zeros(Nit, 1);
for i_it = 1:Nit
    alpha = alpha_ini;
    f(i_it) = Quadratic(A, x);
    nabla = A*x;
%     ฯ฿หั
    while Quadratic(A, x-alpha*nabla) >  (Quadratic(A, x)+alpha*rho*nabla) 
        alpha = alpha/1.2;
    end
    x = x - alpha*nabla;
end
end