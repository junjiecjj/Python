








clc;
clear all;
close all;

rng(42); 


M = 3; 
N = 6;
col_norms = [0.75, 0.75, 1, 1, 1.25, 1.25];
par = 2;
X0 = [0.0748+0.3609i,  0.0392+0.4558i, 0.5648+0.3635i, -0.2567+0.4463i, 0.7064+0.6193i, 0.1586+0.6825i;
     -0.5861-0.0570i, -0.2029+0.8024i, -0.5240+0.4759i, -0.1806-0.1015i, -0.1946-0.1889i, 0.5080+0.0226i;
     -0.7112+0.1076i, -0.2622-0.1921i, -0.1662+0.1416i, 0.0202+0.8316i, 0.0393-0.2060i, 0.2819+0.4135i];

% X0 = [];

parX0 = PAR_cols(X0)

[X, alpha] = AlternatingProjection(M, N, col_norms, par, X0);

fprintf('列范数：\n'); disp(sqrt(sum(abs(X).^2,1))');
fprintf('PAR：\n'); disp(PAR_cols(X));
fprintf('奇异值：\n'); disp(svd(X)');


%% Algorithm 4
c = 3^2; 

d = 4;
z = [zeros(2,1); randn(d,1) + 1i*randn(d,1); zeros(2,1) ; 3+4j; 3+4j];
d = d + 6

d = 20
z = randn(d,1) + 1i*randn(d,1);
d

rho = 1.5;
s = nearestVectorAlgorithm4(z, c, rho);
disp(['Norm squared: ', num2str(norm(s)^2)]);
disp(['PAR: ', num2str(max(abs(s).^2) / (norm(s)^2/d))]);


