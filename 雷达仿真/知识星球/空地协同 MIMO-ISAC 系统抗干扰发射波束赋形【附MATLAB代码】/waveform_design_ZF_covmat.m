function [ R,p,W ] = waveform_design_ZF_covmat( d_theta,H,M,K,P,a,a_theta_bar,theta,SINR_threshold,noise_variance,power )
L = length(theta);
cvx_solver sedumi
cvx_begin quiet
variables bt p(K)
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
H*R*H'==diag(p);
diag(R)==ones(M,1)*power/M;
for k=1:K
    p(k)>=SINR_threshold*noise_variance;
end
cvx_end

[Lr,~] = chol(R);
if size(Lr,1) ~= M
    R = nearestSPD(R);
    [Lr,p] = chol(R);
end
[~,Qh] = qr(H*Lr',0);
W = [Lr*Qh',zeros(M,M)];
end
