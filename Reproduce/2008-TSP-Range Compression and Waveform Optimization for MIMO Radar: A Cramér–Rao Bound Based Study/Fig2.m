

clc;
clear all;
close all;


K = 4;
A = randn(4, 4) + 1j * randn(4, 4);

e2 = zeros(K, 1);
e2(2) = 1;


% e2' * A * conj(e2)
% e2.' * A * e2

