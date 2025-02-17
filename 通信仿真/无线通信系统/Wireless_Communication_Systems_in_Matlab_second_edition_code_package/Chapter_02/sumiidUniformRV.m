function [Y]=sumiidUniformRV(N,k,nSamp) 
% Generate N i.i.d uniformly distributed discrete random variables
% each with nSamp samples in the interval 1:k and sum them up
Y=0;
for i=1:N,
    Y=Y+ceil(k*rand(1,nSamp));
end; end