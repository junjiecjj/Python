function y = remove_cyclic_prefix(r,Ncp,N)
%function to remove cyclic prefix from the received OFDM symbol r
% r - received ofdm symbol with CP 
% Ncp - num. of samples at beginning of r that need to be removed
% N - number of samples in a single OFDM symbol
% y - returns the OFDM symbol without cyclic prefix
y=r(Ncp+1:N+Ncp);%cut from index Ncp+1 to N+Ncp
end