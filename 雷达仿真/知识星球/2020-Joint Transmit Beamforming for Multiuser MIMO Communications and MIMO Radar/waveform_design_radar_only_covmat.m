function [ R ] = waveform_design_radar_only_covmat(d_theta,M,P,a,a_theta_bar,theta,power)
L = length(theta);
cvx_solver sedumi
cvx_begin quiet
variable bt
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
diag(R)==ones(M,1)*power/M;
cvx_end
end
