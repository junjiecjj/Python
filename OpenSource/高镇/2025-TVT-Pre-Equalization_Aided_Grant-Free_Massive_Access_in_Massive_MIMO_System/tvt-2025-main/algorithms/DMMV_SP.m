function [H_hat, supp, iter] = DMMV_SP(Y, Phi, s)
% DMMV-SP: Distributed Multiple Measurement Vector Subspace Pursuit algorithm for DMMV CS problem
% where common support shared by multiple vectors is leveraged.
% An extended version of GSP algorithm in the paper:
% J.Feng et al, "Generalized Subspace Pursuit for Signal Recovery from Multiple-Measurement Vectors"

% Written by Malong Ke (kemalong@bit.edu.cn), Beijing Institute of Technology
% Updated on Jan, 18th 2019

% Inputs£º
%   Y: received signals
%   Phi: measurement matrix
%   s: sparsity level
% Outputs£º
%   H_hat: the estimated sparse matrix
%   supp: support set
%   iter: number of iteration

%% Initializations
[G,M,P] = size(Y);
N = size(Phi,2);   
H_hat = zeros(N,M,P); 
supp_set = []; % support set
r_y = Y;       % residual signals
z_k = s;
iter_max = s;
%% Iterations
for iter = 1:iter_max        
    % Identification
    r_y_pre = r_y; 
    inn_pro = zeros(N,M,P);
    for p = 1:P
        inn_pro(:,:,p) = Phi(:,:,p)'*r_y(:,:,p);
    end
    [~, index_up] = sort(sum(sum(abs(inn_pro),3),2), 'descend');
    index_up = index_up(1:z_k);
    % Support Merger
    supp_set = union(supp_set, index_up); 
    % Least Square Estimation
    Phi_supp = Phi(:,supp_set,:);   
    H_ls = zeros(length(supp_set),M,P);
    for p = 1:P
        H_ls(:,:,p) = Phi_supp(:,:,p)\Y(:,:,p);
    end
    % Support Pruning
    [~, index_pru] = sort(sum(sum(abs(H_ls),3),2), 'descend');
    % Final Support Update
    supp_set = supp_set(index_pru(1:s));
    Phi_supp = Phi(:,supp_set,:);
    H_ls = zeros(length(supp_set),M,P);
    for p = 1:P
        H_ls(:,:,p) = Phi_supp(:,:,p)\Y(:,:,p);
        r_y(:,:,p) = Y(:,:,p) - Phi(:,supp_set,p)*H_ls(:,:,p);
    end
    if G <= s
        z_k = 1;
    else
        z_k = ceil((G-s)/2);
    end
    if norm(r_y(1:end)) > norm(r_y_pre(1:end))  
            break;
    end
end
    H_hat(supp_set,:,:) = H_ls;
    supp = sort(supp_set);
end