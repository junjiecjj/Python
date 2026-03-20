function [x_phi, v_phi, z] = get_Damping_varmatrix(x_phi, v_phi, z, N, beta, v_n, w_0, t)

    for t_ = 1 : t+1
        v_phi(t+1, t_) = (1 / N * z(:, t+1)' * z(:, t_) - beta * v_n) / w_0;
        v_phi(t_, t+1) = v_phi(t+1, t_)';
    end
    
    if v_phi(t+1, t+1) > v_phi(t, t)
        x_phi(:, t+1) = x_phi(:, t);
        v_phi(t+1, t+1) = v_phi(t, t);
        v_phi(1:t, t+1) = v_phi(1:t, t);
        v_phi(t+1, 1:t) = v_phi(t, 1:t);
        z(:, t+1) = z(:, t);
    end


end