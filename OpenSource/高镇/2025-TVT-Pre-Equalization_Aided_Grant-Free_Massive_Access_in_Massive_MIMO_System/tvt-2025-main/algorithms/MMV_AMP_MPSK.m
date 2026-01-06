function [H_amp, lambda, nvar] = MMV_AMP_MPSK(Y, Phi, damp, niter, tol, sel_NN_set, L)
% DMMV-AMP algorithm for DMMV CS problem (estimate 3D matrix),
% where incremental EM algorithm is used to learn unknown hyper-parameters.
% An extended version of AMP-NNSPL algorithm in the paper:
% X. Meng et al, "Approximate message passing with nearest neighbor sparsity pattern learning".

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Omega=pskmod([0:L-1], L, 0, 'gray');% using your psk modulation func here
i_Omega=1:L; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
snr0 = 100;
[M,Q] = size(Y);
P = 1;
[~,N,~] = size(Phi);
alpha = M/N;
normal_cdf = @(x) 1/2.*(1+erf(x/sqrt(2)));
normal_pdf = @(x) 1/sqrt(2*pi).*exp(-x.^2/2);
alpha_grid = linspace(0,10,1024);
rho_SE = (1 - (2/alpha).*((1+alpha_grid.^2).*normal_cdf(-alpha_grid)-alpha_grid.*normal_pdf(alpha_grid)))...
         ./ (1 + alpha_grid.^2 - 2*((1+alpha_grid.^2).*normal_cdf(-alpha_grid)-alpha_grid.*normal_pdf(alpha_grid)));
lambda = 0.1*max(rho_SE).*ones(N,Q,P);  % belief indicators
% lambda = 0.5.*ones(N,Q,P);  % belief indicators
xmean = zeros(N,Q,P);
xvar = zeros(N,Q,P);
nvar0 = zeros(M,Q,P); 
for p = 1:P
    for q = 1:Q
        nvar0(:,q,p) = norm(Y(:,q,p))^2/(1+snr0)/M;
        xvar(:,q,p) = (norm(Y(:,q,p))^2-M*nvar0(1,q,p))/(norm(Phi(:,:,p),'fro')^2);
    end
end
nvar = sum(nvar0(:))/P/Q;
% nvar(:,:,:) = sum(nvar(:))/P/Q;
% xvar(:,:,:) = sum(xvar(:))/P/Q;
% lambda = 0.5.*ones(N,Q,P);  % belief indicators
% xmean = zeros(N,Q,P);
% xvar = ones(N,Q,P);
% nvar = ones(M,Q,P); 
% index for nearest neighbor sparsity learning 
index = ones(N,Q,P);
% index_up = [zeros(1,Q,P); index(1:N-1,:,:)];
% index_down = [index(2:N,:,:); zeros(1,Q,P)];
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
nvartmp = zeros(N,Q,P);
Z_0 = zeros(N,Q,P);
Z_1 = zeros(N,Q,P);
Z_2 = zeros(N,Q,P);
%% AMP iterations
for iter = 1:niter
    H_amp_pre = H_amp;
    V_pre = V;
    for p = 1:P
        % factor node update
        V(:,:,p) = damp.*V_pre(:,:,p) + (1-damp).*abs(Phi(:,:,p)).^2*v(:,:,p);
        Z(:,:,p) = damp.*Z(:,:,p) + (1-damp).*(Phi(:,:,p)*H_amp(:,:,p)-(Y(:,:,p)-Z(:,:,p))./(nvar+V_pre(:,:,p)).*V(:,:,p));
        % variable node update 
        D(:,:,p) = 1 ./ ((abs(Phi(:,:,p)).^2).'*(1./(nvar+V(:,:,p))));
        C(:,:,p) = H_amp(:,:,p) + D(:,:,p).*(Phi(:,:,p)'*((Y(:,:,p)-Z(:,:,p))./(nvar+V(:,:,p))));
        % compute the estimated channels and its variance
%         L_cal(:,:,p) = (1/2).*(log(D(:,:,p)./(D(:,:,p)+xvar(:,:,p))) + abs(C(:,:,p)).^2./D(:,:,p) - abs(C(:,:,p)-xmean(:,:,p)).^2./(D(:,:,p)+xvar(:,:,p))); 
%         pai(:,:,p) = lambda(:,:,p) ./ (lambda(:,:,p)+(1-lambda(:,:,p)).*exp(-L_cal(:,:,p))); 
%         A(:,:,p) = (xvar(:,:,p).*C(:,:,p)+xmean(:,:,p).*D(:,:,p)) ./ (D(:,:,p)+xvar(:,:,p));
%         B(:,:,p) = (xvar(:,:,p).*D(:,:,p)) ./ (xvar(:,:,p)+D(:,:,p));
%         H_amp(:,:,p) = pai(:,:,p).*A(:,:,p);
%         v(:,:,p) = pai(:,:,p).*(abs(A(:,:,p)).^2+B(:,:,p)) - abs(H_amp(:,:,p)).^2;
        for j=1:Q
            tao_new = mean(C(:,j,p),1);
            Z_0(:,j,p)=sum(lambda(:,j,p)/L.*exp(-(abs(Omega(i_Omega)).^2-2*real(conj(C(:,j,p))*Omega(i_Omega)))./D(:,j,p)),2);
            Z_1(:,j,p)=sum(lambda(:,j,p)/L*Omega(i_Omega).*exp(-(abs(Omega(i_Omega)).^2-2*real(conj(C(:,j,p))*Omega(i_Omega)))./D(:,j,p)),2);
            Z_2(:,j,p)=sum(lambda(:,j,p)/L*abs(Omega(i_Omega)).^2.*exp(-(abs(Omega(i_Omega)).^2-2*real(conj(C(:,j,p))*Omega(i_Omega)))./D(:,j,p)),2);
        end
        pai=1./(1+max((1-lambda)./Z_0, 1e-15));       %(15)-Mei
        xi=Z_1./((1-lambda)+Z_0);                    %(17)-Mei
        v(:,:,p)=Z_2./((1-lambda)+Z_0)-abs((xi)).^2; %(18)-Mei
        H_amp(:,:,p) = xi;
 
        for q = 1:Q
%             xmean(:,q,p) = sum(pai(:,q,p).*A(:,q,p))/sum(pai(:,q,p));
%             xvar(:,q,p) = sum(pai(:,q,p).*(abs(xmean(:,q,p)-A(:,q,p)).^2+B(:,q,p)))/sum(pai(:,q,p)); 
            nvartmp(:,q,p) = sum(abs(Y(:,q,p)-Z(:,q,p)).^2./abs(1+V(:,q,p)./nvar).^2 + V(:,q,p)./(1+V(:,q,p)./nvar))/M;
        end
    end
    nvar = sum(nvartmp(:))/Q/N/P;
    
    % nearest neighbor sparsity pattern learning
    if sel_NN_set == 0
       pai_temp = sum(sum(pai,3),2)./Q./P;
       lambda = pai_temp(:,ones(1,Q),ones(1,P)); 
%        lambda = pai;
    else
%        pai_up = [zeros(1,Q,P); pai(1:N-1,:,:)];
%        pai_down = [pai(2:N,:,:); zeros(1,Q,P)];
       pai_left = [zeros(N,1,P), pai(:,1:Q-1,:)];
       pai_right = [pai(:,2:Q,:), zeros(N,1,P)];
       pai_ahead = cat(3, zeros(N,Q,1), pai(:,:,1:P-1));
       pai_latter = cat(3, pai(:,:,2:P), zeros(N,Q,1));
       lambda = (pai_left + pai_right + pai_ahead + pai_latter)./index_3D; % pai_up + pai_down + 
    end
   
    % stopping criteria
    if norm(H_amp_pre(:)-H_amp(:))/norm(H_amp(:)) < tol
       break;
    end 
    
end
end