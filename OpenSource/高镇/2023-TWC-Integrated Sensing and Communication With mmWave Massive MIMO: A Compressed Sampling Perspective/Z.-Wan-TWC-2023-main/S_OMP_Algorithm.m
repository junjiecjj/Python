function [h_v_hat, iter_num, Support_index_set, xi_hat_com] = S_OMP_Algorithm(y_k_com, Q, epsilon, iter_max)
% S-OMP，兼容SMV-OMP

[M, K] = size(y_k_com);    % size of y_k_com, K是MMV的数量，可为1（SMV）
N = size(Q,2);

% Initialize the residual vectors to the input signal vectors and support estimate, where ..._com contain K columns
Support_index_set = [];
r_k_com = y_k_com;
MSE = epsilon+rand;          % Pre-define MSE
iter_num = 0;       % Initialize the number of iteration
h_v_hat = zeros(N,K);

while (MSE > epsilon && iter_num < iter_max)
    % Distributed Correlation
    c_k_com = Q'*r_k_com;
    
    % Find the maximum projection along the different spaces
    [~, index_p] = max(sum(abs(c_k_com),2));
    
    % Update the current guess of the common support
    Support_index_set = [Support_index_set; index_p];
    
    % Project the input signal onto the subspace given by the support using WLS
    xi_hat_com = Q(:,Support_index_set)\y_k_com;
    
    % Update residual
    r_k_com = y_k_com - Q(:,Support_index_set)*xi_hat_com;
    
    % Compute the current MSE
    MSE = 1/(M*K)*trace(r_k_com'*r_k_com);
    
    % Compte the number of iteration
    iter_num = iter_num + 1;
end

% assign estimated complex channel gains to the sparse vector
h_v_hat(Support_index_set,:) = xi_hat_com;

end