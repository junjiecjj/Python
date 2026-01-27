function nabla_a = Nabla_a(A, x, alpha)
    nabla  = Nabla(A, x);
    nabla_a = -nabla'*A*x+alpha*(nabla'*A*nabla);
end