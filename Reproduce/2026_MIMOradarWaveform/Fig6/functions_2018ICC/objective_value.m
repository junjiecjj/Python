

% Eq.(12a)

function obj = objective_value(H, S, X, X0, rho)
    obj = rho * norm(H * X - S, 'fro')^2 + (1 - rho) * norm(X - X0, 'fro')^2;
end