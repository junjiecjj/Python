function f = GD_Opt(Nit, A, x)
f = zeros(Nit, 1);
for i_it = 1:Nit
    f(i_it) = Quadratic(A, x);
    nabla = A*x;
    % 最优步长
    alpha = (nabla'*A*x)/(nabla'*A*nabla);
%     fprintf('i_it=%d\n', i_it);
%     fprintf('alpha=%f\n', alpha);
    x = x - alpha*nabla;
end    
end