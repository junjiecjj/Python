


clc;
clear all;
close all;
addpath('./functions');
rng(42);

rng(1);
N = 4;
K = 3;
L = 5;
PT = 1;
rho = 0.3;
epsTol = 1e-10;
maxIter = 200;
H = (randn(K, N) + 1j * randn(K, N)) / sqrt(2);
data = randi([0, 3], K, L);
S = pskmod(data, 4, pi / 4, 'gray');
X0Raw = (randn(N, L) + 1j * randn(N, L)) / sqrt(2);
X0 = sqrt(L * PT) * X0Raw / norm(X0Raw, 'fro');
XAlg = algorithm1_tradeoff(H, S, X0, PT, rho);
XCvx = cvx_problem12_sdr(H, S, X0, PT, rho);

powerTarget = L * PT;
objAlg = rho * norm(H * XAlg - S, 'fro')^2 + (1 - rho) * norm(XAlg - X0, 'fro')^2;
objCvx = rho * norm(H * XCvx - S, 'fro')^2 + (1 - rho) * norm(XCvx - X0, 'fro')^2;

powerAlg = norm(XAlg, 'fro')^2;
powerCvx = norm(XCvx, 'fro')^2;

fprintf('Comparison between Algorithm 1 and CVX-SDR:\n');
fprintf('N                      = %d\n', N);
fprintf('K                      = %d\n', K);
fprintf('L                      = %d\n', L);
fprintf('rho                    = %.4f\n', rho);
fprintf('\n');
fprintf('Algorithm 1 result:\n');
fprintf('objective              = %.12e\n', objAlg);
fprintf('power                  = %.12e\n', powerAlg);
 
fprintf('\n');
fprintf('CVX-SDR result:\n');
fprintf('objective              = %.12e\n', objCvx);
fprintf('power                  = %.12e\n', powerCvx);
 




















