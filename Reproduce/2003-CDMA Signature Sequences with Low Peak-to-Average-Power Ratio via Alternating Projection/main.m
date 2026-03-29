
clc;
clear all;
close all;

rng(42); 


% M = 3; 
% N = 6;
% col_norms = [0.75, 0.75, 1, 1, 1.25, 1.25];
% par = 2;
% X0 = [0.0748+0.3609i,  0.0392+0.4558i, 0.5648+0.3635i, -0.2567+0.4463i, 0.7064+0.6193i, 0.1586+0.6825i;
%      -0.5861-0.0570i, -0.2029+0.8024i, -0.5240+0.4759i, -0.1806-0.1015i, -0.1946-0.1889i, 0.5080+0.0226i;
%      -0.7112+0.1076i, -0.2622-0.1921i, -0.1662+0.1416i, 0.0202+0.8316i, 0.0393-0.2060i, 0.2819+0.4135i];
% 
% % X0 = [];
% 
% parX0 = PAR_cols(X0)
% 
% X = AlternatingProjection(M, N, col_norms, par, X0);
% 
% fprintf('列范数：\n'); disp(sqrt(sum(abs(X).^2,1)));
% fprintf('PAR：\n'); disp(PAR_cols(X));
% fprintf('奇异值：\n'); disp(svd(X)');

M = 3; 
N = 6;
X1 = [
    0.1345 + 0.5625i,  0.1672 + 0.5526i,  0.4439 + 0.3692i, -0.3358 + 0.4696i,  0.4737 + 0.3300i,  0.0944 + 0.5696i;
    0.5410 - 0.2017i, -0.0303 + 0.5766i, -0.5115 + 0.2679i, -0.5432 - 0.1956i, -0.3689 - 0.4442i,  0.5747 + 0.0554i;
   -0.5768 + 0.0252i, -0.2777 - 0.5062i, -0.2303 + 0.5294i,  0.1258 + 0.5635i, -0.0088 - 0.5773i,  0.4132 + 0.4033i
    ];
d = size(X1, 1);  % 3
col_norms = sqrt(sum(abs(X1).^2, 1));  % 应为 1
rho = max(abs(X1).^2, [], 1) ./ (mean(abs(X1).^2, 1));  % 应为 1
sv = svd(X1);  % 应为 [sqrt(2); sqrt(2); sqrt(2)]? 因为 M=3, N=6, 列范数全为1，则 α = 1/3 * sum(1^2) = 2，奇异值应为 sqrt(2) ≈ 1.4142
disp('列范数：'); disp(col_norms);
disp('PAR：'); disp(rho);
disp('奇异值：'); disp(sv);

rho = 2;
col_norms = [0.75, 0.75, 1, 1, 1.25, 1.25];
X = AlternatingProjection(M, N, col_norms, rho, X1);

fprintf('列范数：\n'); disp(sqrt(sum(abs(X).^2,1)));
fprintf('PAR：\n'); disp(PAR_cols(X));
fprintf('奇异值：\n'); disp(svd(X)');



%% Algorithm 4
c = 0.75^2; 

d = 4;
z = [zeros(5,1); randn(4,1) + 1i*randn(4,1); 3+4j; 3+4j; 3+4j];
d = size(z,1);

% d = 9;
% z = [zeros(d,1); 3+4j; 3+4j; 3+4j];
% d = size(z,1);

% d = 8;
% z = randn(d,1) + 1i*randn(d,1);

rho = 2;
[s, k] = nearestVectorAlgorithm4(z, c, rho);
fprintf('输出 s:\n'); disp(s);
fprintf('范数平方: %f (应等于 %f)\n', norm(s)^2, c);
fprintf('PAR: %f (应 ≤ %f)\n', max(abs(s).^2) / (norm(s)^2/d), rho);
fprintf('\n');

if 0
    % 测试用例1：保留集全为零分支
    d1 = 5;
    c1 = 1;
    rho1 = 2;
    % 构造未归一化的 z，归一化后前三个零，后两个幅度为 0.5 和 0.8660
    z1 = [0; 0; 0; 0.5; 0.8660];
    s1 = nearestVectorAlgorithm4(z1, c1, rho1);
    fprintf('=== 测试用例1：保留集全为零分支 ===\n');
    fprintf('输入 z (归一化后):\n'); disp(z1);
    fprintf('输出 s:\n'); disp(s1);
    fprintf('范数平方: %f (应等于 %f)\n', norm(s1)^2, c1);
    fprintf('PAR: %f (应 ≤ %f)\n', max(abs(s1).^2) / (norm(s1)^2/d1), rho1);
    fprintf('前三个分量是否非零？ %d\n', all(abs(s1(1:3)) > 1e-6));
    fprintf('\n');
    
    % 测试用例2：正常分支（保留集非全为零）
    d2 = 3;
    c2 = 1;
    rho2 = 2;
    % 构造未归一化的 z，归一化后幅度为 0.6, 0.8, 0
    z2 = [0.6; 0.8; 0];
    s2 = nearestVectorAlgorithm4(z2, c2, rho2);
    fprintf('=== 测试用例2：正常分支 ===\n');
    fprintf('输入 z (归一化后):\n'); disp(z2);
    fprintf('输出 s:\n'); disp(s2);
    fprintf('范数平方: %f (应等于 %f)\n', norm(s2)^2, c2);
    fprintf('PAR: %f (应 ≤ %f)\n', max(abs(s2).^2) / (norm(s2)^2/d2), rho2);
    
    
    % 测试用例3：all 0分支（保留集非全为零）
    d2 = 3;
    c2 = 1;
    rho2 = 2;
    % 构造未归一化的 z，归一化后幅度为 0.6, 0.8, 0
    z2 = [0.0; 0.0; 0];
    s2 = nearestVectorAlgorithm4(z2, c2, rho2);
    fprintf('=== 测试用例2：正常分支 ===\n');
    fprintf('输入 z (归一化后):\n'); disp(z2);
    fprintf('输出 s:\n'); disp(s2);
    fprintf('范数平方: %f (应等于 %f)\n', norm(s2)^2, c2);
    fprintf('PAR: %f (应 ≤ %f)\n', max(abs(s2).^2) / (norm(s2)^2/d2), rho2);
end