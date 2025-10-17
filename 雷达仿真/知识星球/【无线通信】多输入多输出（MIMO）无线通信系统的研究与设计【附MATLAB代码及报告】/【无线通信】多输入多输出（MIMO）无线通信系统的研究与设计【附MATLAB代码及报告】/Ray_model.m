function H=Ray_model(L)
% Rayleigh channel model 
% Input : L = Number of channel realizations
% Output: H = Channel vector
H = (randn(1,L)+j*randn(1,L))/sqrt(2);