function [h_v_hat, iter_num, Support_index_set, BlkSupp_index_set] = B_OMP_Algorithm(y, Q, blk_len, epsilon, iter_max, Ts, tau_pulse)
% Block-OMP

M = length(y);
N = size(Q,2);

% Initialize the residual vectors to the input signal vectors and support estimate, where ..._com contain K columns
Support_index_set = [];
BlkSupp_index_set = [];
r = y;
MSE = epsilon+rand;          % Pre-define MSE
iter_num = 0;       % Initialize the number of iteration
h_v_hat = zeros(N,1);

while (MSE > epsilon && length(BlkSupp_index_set) < iter_max)
    % Distributed Correlation
    c = Q'*r;
    c_blk = zeros(N/blk_len,1);
    for i = 1:N/blk_len
        c_blk(i) = norm(c((i-1)*blk_len+1:i*blk_len),2);
    end
    
    % Find the maximum projection along the different spaces
    [~, index_p] = max(c_blk);    
    
    % Update the current guess of the common support
    Support_index_set = [Support_index_set, (index_p-1)*blk_len+1:index_p*blk_len];
    if((index_p-1)*Ts-tau_pulse >= 0)
        BlkSupp_index_set = union(BlkSupp_index_set, index_p);
    end
    
    % Project the input signal onto the subspace given by the support using WLS
    xi_hat_com = Q(:,Support_index_set)\y;
    
    % Update residual
    r = y - Q(:,Support_index_set)*xi_hat_com;
    
    % Compute the current MSE
    MSE = 1/(M)*trace(r'*r);
    
    % Compte the number of iteration
    iter_num = iter_num + 1;
end

% assign estimated complex channel gains to the sparse vector
h_v_hat(Support_index_set,:) = xi_hat_com;

end