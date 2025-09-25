function [obj] = Compute_obj(sigma_B,sigma_E,p,H,h)
% input:                sigma_B: noise power at users with eavesdroppers (Bob)  -90dBm
%                       sigma_E: noise power at eavesdroppers (Eve)  -80dBm
%                       p: power allocation
%                       H: direct channel of legimate users
%                       h: direct channel of eavesdroppers
%                       weight: weight of data rates of users without eavesdroppers
% output:               obj: sum rates
N_B = size(sigma_B,1); 
Secure_R = zeros(N_B,1); 
for n = 1:N_B
    Secure_R(n) = log2(1+(H(n,n)*p(n))/((H(n,:)*p-H(n,n)*p(n))+sigma_B(n)))-log2(1+(h(n,n)*p(n))/((h(n,:)*p-h(n,n)*p(n))+sigma_E(n)));
end
obj = sum(Secure_R);
end