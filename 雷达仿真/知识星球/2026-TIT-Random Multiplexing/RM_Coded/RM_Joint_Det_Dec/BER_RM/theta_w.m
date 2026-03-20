%% theta_(i) * w(j)
function res = theta_w(lambda_s, B, sign, log_B, log_theta_i, j, N)
    tmp = (lambda_s - B) .* sign.^j .* exp(log_theta_i + j * log_B);
    res = 1 / N * sum(tmp); 
end