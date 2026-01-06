function [H_hat, supp_set, iter_num] = DMMV_OMP(Y, Phi, epsilon)
% DMMV-OMP: Distributed Multiple Measurement Vector Orthogonal Matching Pursuit algorithm for DMMV CS problem
% where common support shared by multiple vectors is leveraged.
% An extended version of SOMP algorithm in the paper:
% J.Determe, et al, "On the Noise Robustness of Simultaneous Orthogonal Matching Pursuit"

% Written by Malong Ke (kemalong@bit.edu.cn), Beijing Institute of Technology
% Version: 2019.01.18

% Inputs£º
%   Y: received signals
%   Phi: measurement matrix
%   epsilon: a tunable parameter defining the maximum error between the eatimate and the received signal
% Outputs£º
%   H_hat: the estimated sparse matrix
%   iter_num: number of iteration

%% Initializations
[G, M, P] = size(Y);   
N = size(Phi,2);
H_hat = zeros(N,M,P);
supp_set = [];
r_y = Y;
MSE = 2*epsilon; % Pre-define MSE
iter_num = 0;    % Initialize the number of iteration
%% Iterations
while (MSE > epsilon ) 
    % Distributed Correlation
    inn_pro = zeros(N,M,P);
    for p = 1:P
        inn_pro(:,:,p) = Phi(:,:,p)'*r_y(:,:,p);
    end
    % Find the maximum projection along the different spaces
    [~, index_up] = max(sum(sum(abs(inn_pro),3),2));
    % Update the current guess of the common support
    supp_set = union(supp_set, index_up);
    % Project the input signal onto the subspace given by the support using LS
    Phi_supp = Phi(:,supp_set,:);  
    H_ls = zeros(length(supp_set),M,P);
    MSE = 0;
    for p = 1:P
        H_ls(:,:,p) = Phi_supp(:,:,p)\Y(:,:,p);
        r_y(:,:,p) = Y(:,:,p) - Phi(:,supp_set,p)*H_ls(:,:,p); % Update residual
        MSE = MSE + trace(r_y(:,:,p)'*r_y(:,:,p));
    end  
    MSE = MSE/(G*M*P); % Compute the current MSE
    iter_num = iter_num + 1;       % Compte the number of iteration
end
% assign estimated complex channel gains to the sparse vector
H_hat(supp_set,:,:) = H_ls;
end