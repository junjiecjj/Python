function [sumAoI,A] = sum_priority_AoI_obj(mu,rho)
% input:  mu:     service rate
%         rho:    arrival rate
% output: sumAoI: sum of each source's AoI
%         A:      each source's AoI
n = size(rho,1);% number of sources
A = zeros(n,1);
for i = 1:n
    A(i) = ((ones(1,i-1)*rho(1:i-1))^2+3*(ones(1,i-1)*rho(1:i-1))+1)/(mu*(ones(1,i-1)*rho(1:i-1)+1))+(((ones(1,i-1)*rho(1:i-1)+1)^2))/(mu*rho(i));
end
sumAoI = sum(A);
end