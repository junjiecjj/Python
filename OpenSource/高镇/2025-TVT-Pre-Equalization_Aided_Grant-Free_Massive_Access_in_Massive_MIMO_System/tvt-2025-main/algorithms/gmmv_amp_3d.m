function [H_amp, lambda] = gmmv_amp_3d(Y, Phi, damp, niter, tol, sel_NN_set)
% GMMV-AMP algorithm for GMMV CS problem (estimate 3D matrix),
% where incremental EM algorithm is used to learn unknown hyper-parameters.

% Written by Malong Ke (kemalong@bit.edu.cn), Beijing Institute of Technology
% version: 2019.03.14

% Inputs��
%   Y: received signals
%   Phi: measurement matrix
%   niter: number of AMP iterations
%   tol: termination threshold
%   sel_NN_sel: % select nearest neighbor set for NNSPL 0: strctured sparsity   1: clustered sparsity
% Outputs��
%   H_amp: the estimated channels
%   lambda: belief indicators

%% Initializations
% hyper-parameters initialization
snr0 = 100;
[M,Q,P] = size(Y);
[~,N,~] = size(Phi);
alpha = M/N;
normal_cdf = @(x) 1/2.*(1+erf(x/sqrt(2)));
normal_pdf = @(x) 1/sqrt(2*pi).*exp(-x.^2/2);
alpha_grid = linspace(0,10,1024);
rho_SE = (1 - (2/alpha).*((1+alpha_grid.^2).*normal_cdf(-alpha_grid)-alpha_grid.*normal_pdf(alpha_grid)))...
         ./ (1 + alpha_grid.^2 - 2*((1+alpha_grid.^2).*normal_cdf(-alpha_grid)-alpha_grid.*normal_pdf(alpha_grid)));
lambda = alpha*max(rho_SE).*ones(N,Q,P);  % belief indicators
% lambda = 0.5.*ones(N,Q,P);  % belief indicators
xmean = zeros(N,Q,P);
xvar = zeros(N,Q,P);
nvar = zeros(M,Q,P); 
for p = 1:P
    for q = 1:Q
        nvar(:,q,p) = norm(Y(:,q,p))^2/(1+snr0)/M;
        xvar(:,q,p) = (norm(Y(:,q,p))^2-M*nvar(1,q,p))/(norm(Phi(:,:,p),'fro')^2);
    end
end
% index for nearest neighbor sparsity learning 
index = ones(N,Q,P);
index_left = [zeros(N,1,P), index(:,1:Q-1,:)];
index_right = [index(:,2:Q,:), zeros(N,1,P)];
index_ahead = cat(3, zeros(N,Q,1), index(:,:,1:P-1));
index_latter = cat(3, index(:,:,2:P), zeros(N,Q,1));
index_3D = index_left + index_right + index_ahead + index_latter; % index_up + index_down + 
clear index
clear index_left
clear index_right
clear index_ahead
clear index_latter
% other parameters initialization
H_amp = xmean; 
v = xvar;
V = ones(M,Q,P);
Z = Y;
% preallocate the memory
D = zeros(N,Q,P);
C = zeros(N,Q,P);
L_cal = zeros(N,Q,P);
pai = zeros(N,Q,P);
A = zeros(N,Q,P);
B = zeros(N,Q,P);

%% AMP iterations
for iter = 1:niter
    H_amp_pre = H_amp;
    V_pre = V;
    for p = 1:P
        % factor node update
        V(:,:,p) = damp.*V_pre(:,:,p) + (1-damp).*abs(Phi(:,:,p)).^2*v(:,:,p);
        Z(:,:,p) = damp.*Z(:,:,p) + (1-damp).*(Phi(:,:,p)*H_amp(:,:,p)-(Y(:,:,p)-Z(:,:,p))./(nvar(:,:,p)+V_pre(:,:,p)).*V(:,:,p));
        % variable node update 
        D(:,:,p) = 1 ./ ((abs(Phi(:,:,p)).^2).'*(1./(nvar(:,:,p)+V(:,:,p))));
        C(:,:,p) = H_amp(:,:,p) + D(:,:,p).*(Phi(:,:,p)'*((Y(:,:,p)-Z(:,:,p))./(nvar(:,:,p)+V(:,:,p))));
        % compute the estimated channels and its variance
        L_cal(:,:,p) = (log(D(:,:,p)./(D(:,:,p)+xvar(:,:,p))) + abs(C(:,:,p)).^2./D(:,:,p) - abs(C(:,:,p)-xmean(:,:,p)).^2./(D(:,:,p)+xvar(:,:,p))); 
        pai(:,:,p) = lambda(:,:,p) ./ (lambda(:,:,p)+(1-lambda(:,:,p)).*exp(-L_cal(:,:,p))); 
        A(:,:,p) = (xvar(:,:,p).*C(:,:,p)+xmean(:,:,p).*D(:,:,p)) ./ (D(:,:,p)+xvar(:,:,p));
        B(:,:,p) = (xvar(:,:,p).*D(:,:,p)) ./ (xvar(:,:,p)+D(:,:,p));
        H_amp(:,:,p) = pai(:,:,p).*A(:,:,p);
        v(:,:,p) = pai(:,:,p).*(abs(A(:,:,p)).^2+B(:,:,p)) - abs(H_amp(:,:,p)).^2;
        for q = 1:Q
            xmean(:,q,p) = sum(pai(:,q,p).*A(:,q,p))/sum(pai(:,q,p));
            xvar(:,q,p) = sum(pai(:,q,p).*(abs(xmean(:,q,p)-A(:,q,p)).^2+B(:,q,p)))/sum(pai(:,q,p)); 
            nvar(:,q,p) = sum(abs(Y(:,q,p)-Z(:,q,p)).^2./abs(1+V(:,q,p)./nvar(:,q,p)).^2 + V(:,q,p)./(1+V(:,q,p)./nvar(:,q,p)))/M;
        end
    end
    
    % nearest neighbor sparsity pattern learning
    if sel_NN_set == 0
       pai_temp = sum(sum(pai,3),2)./Q./P;
       lambda = pai_temp(:,ones(1,Q),ones(1,P)); 
    else
       pai_left = [zeros(N,1,P), pai(:,1:Q-1,:)];
       pai_right = [pai(:,2:Q,:), zeros(N,1,P)];
       pai_ahead = cat(3, zeros(N,Q,1), pai(:,:,1:P-1));
       pai_latter = cat(3, pai(:,:,2:P), zeros(N,Q,1));
       lambda = (pai_left + pai_right + pai_ahead + pai_latter)./index_3D; 
    end
   
    % stopping criteria
    if norm(H_amp_pre(:)-H_amp(:))/norm(H_amp(:)) < tol
       break;
    end 
    
end
end