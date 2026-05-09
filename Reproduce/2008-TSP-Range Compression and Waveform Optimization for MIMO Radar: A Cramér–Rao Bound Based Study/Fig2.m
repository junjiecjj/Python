

clc;
clear all;
close all;


K = 6;
A = randn(K, K) + 1j * randn(K, K);

e2 = zeros(K, 1);
e2(2) = 1;



% e2' * A * conj(e2)
e2.' * A * e2

