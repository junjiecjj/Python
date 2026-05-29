


clc;
clear all;
close all;
addpath('./functions');
rng(42);
p.K = 4;                            % # of Users
p.N = 16;                           % # of Antennas per Each Users (ULA)
p.L = 20;                           % # of Communication Frame
p.Pt = 1;                           % Total Power Constraint
p.N0dB = 2 : -2 : -12;              % Noise Settings
p.N0 = 10.^(p.N0dB ./ 10);  
p.SNR = p.Pt ./ p.N0;
p.SNRdB = 10 * log10(p.SNR);
% Radar Settings
p.theta = -pi/2 : pi/180 : pi/2;        % Radar ULA Angle Settings
p.theta_target = [0];
p.target_DoA = [0];

p.beam_width= 9;
p.l=ceil((p.target_DoA + pi/2 * ones(1, length(p.target_DoA)))/(pi/180) + ones(1, length(p.target_DoA)));
p.Pd_theta = zeros(length(p.theta), 1);

for idx = 1:length(p.target_DoA)
    p.Pd_theta(p.l(idx)-(p.beam_width-1)/2 : p.l(idx)+(p.beam_width-1)/2, 1) = ones(p.beam_width, 1);
end

rng(1);
N = 4;
K = 4;
L = 20;
PT = 1;
rho = 0.1;
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
 




















