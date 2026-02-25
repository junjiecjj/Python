function x = newtonOptimization(x_, t, A, Pdesired, ang)
    epsilon = 1e-3;

    % Parameters for Armijo rule
    s = 2;
    beta = 0.5;
    sigma = 0.1;
    
    stopCriteriaMet = false;
    J_ = squaredErrorObjective(x_, t, A, Pdesired, ang);
    
    while ~stopCriteriaMet
        [g, H] = gradientAndHessian(x_, t, A, Pdesired, ang);

        % Descent direction
        d = -(H\g);

        % Compute step size and the new value x using the Armijo rule
        m = 0;
        gamma = g'*d;
        while true 
            mu = (beta^m)*s;
            x = x_ + mu*d;
    
            J = squaredErrorObjective(x, t, A, Pdesired, ang);

            if abs(J_) - abs(J) >= (-sigma*mu*gamma)
                x_ = x;
                stopCriteriaMet = abs(J - J_) < epsilon;
                J_ = J;
                break;
            end
            m = m + 1;
        end
    end
end