function [ R,r,W ] = waveform_design_SDR_covmat( d_theta,H,M,K,P,a,a_theta_bar,theta,SINR_threshold,noise_variance,power )
L = length(theta);
t = zeros(M,M);
Wc = zeros(M,K);
cvx_solver sedumi
cvx_begin quiet
variable bt
variable r(M,M,K) hermitian semidefinite
variable R(M,M) hermitian semidefinite
expressions u1(L) u2((P.^2-P)/2)
for ii=1:L
    u1(ii)=(bt*d_theta(ii)-a(:,ii)'*R*a(:,ii));
end

for ii=1:P-1
    for jj=ii+1:P
        u2(ii+jj-2) = a_theta_bar(:,jj)'*R*a_theta_bar(:,ii);
    end
end

minimize square_pos(norm(u1,2))/L + square_pos(norm(u2,2))*(2./(P.^2-P))
subject to
    R - sum(r,3) == hermitian_semidefinite(M);
    diag(R) == ones(M,1)*power/M;
    for k=1:K
        real((1+1/SINR_threshold)*H(k,:)*r(:,:,k)*H(k,:)') >= real(H(k,:)*R*H(k,:)'+noise_variance);
    end
cvx_end
for k=1:K
    wk = (H(k,:)*r(:,:,k)*H(k,:)')^(-1/2)*r(:,:,k)*H(k,:)';
    r(:,:,k) = wk*wk';
    Wc(:,k) = wk;
end
Wr_WrH = R - sum(r,3);
[Wr,p] = chol(Wr_WrH);
if size(Wr,1) ~= M
    Wr_WrH = nearestSPD(Wr_WrH);
    [Wr,p] = chol(Wr_WrH);
end
W = [Wc,Wr'];
end
