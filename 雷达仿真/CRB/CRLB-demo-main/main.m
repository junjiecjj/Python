%% 
clear;
clc;
close all;
addpath('./func')
rng(666);
%% 参数设置
N               =           64;                % 基站处天线
fc              =           100e9;              %100GHz
lambda_c        =           3e8/fc;
d               =           lambda_c/2;
T               =           10;
Nit             =           1e2;
SNRdBs          =           -10:5:20;
%% 
theta   = 0.5;%pi/6
H   = genSteerVector(theta, N, d, lambda_c);
Y   = zeros(N, T);
A   = eye(N);
a1  = genPartialSteerVector(theta, N, d, lambda_c, 1);
a2  = genPartialSteerVector(theta, N, d, lambda_c, 2);
MseMUSIC    = zeros(length(SNRdBs), 1);
MseESPRIT   = zeros(length(SNRdBs), 1);
CRLB        = zeros(length(SNRdBs), 1);
D           = a1;
Cst         = D'*(eye(N)-H*inv(H'*H)*H')*D;
for i_SNR = 1:length(SNRdBs)
    fprintf('==============================================\n');
    fprintf('SNR        = %ddB\n', SNRdBs(i_SNR));
    for i_it = 1:Nit
        
        if mod(i_it,10) == 0
            fprintf('i_it/Nit   = %3d/%3d\n', i_it, Nit);
        end
        X = (sqrt(2)/2*randn(1,T)+1j*sqrt(2)/2*randn(1,T));
        for i_T = 1:T
            Y(:, i_T) = awgn(H*X(i_T), SNRdBs(i_SNR), 'measured');
        end
        
        theta_MUISC     = MUSIC(Y);
        psi = TLS_ESPRIT_Algorithm(Y, 1);
        theta__ESPRIT = log(psi)/(1j*pi); 
        MseMUSIC(i_SNR)     = MseMUSIC(i_SNR) + abs(asin(theta_MUISC) - asin(theta))^2;
        MseESPRIT(i_SNR)    = MseESPRIT(i_SNR) + abs(asin(theta__ESPRIT) - asin(theta))^2;
        
        sigma2  = 10^(-SNRdBs(i_SNR)/10);
% 这个sigma2和上面的值是渐进一致的
%         sigma2  = (norm(Y, 'fro')^2-norm(H*X, 'fro')^2)/...
%                 (size(Y, 1)*size(Y, 2));
%         X_bar   = kron(X.', eye(N));
%         y       = reshape(Y,[],1);
%         CRLB(i_SNR)     = CRLB(i_SNR) + sigma2/2./...
%                             real(-y'*X_bar*a2 + a2'*(X_bar'*X_bar)*H + a1'*(X_bar'*X_bar)*a1);
        CRLB(i_SNR)     = CRLB(i_SNR) + sigma2/2/real((Cst*(X*X')));
    end
end
MseMUSIC     = MseMUSIC/Nit;
MseESPRIT    = MseESPRIT/Nit;
CRLB         = CRLB/Nit;

% fprintf("计算得到的SNRdB = %f\n", 10*log10((norm(H*X, 'fro')^2)/(norm(Y, 'fro')^2-norm(H*X, 'fro')^2)))
%% 绘图
set(0,'defaultfigurecolor','w') 
figure; hold on; grid on; box on;
xlabel('SNR/dB');
ylabel('MSE(rad^2)');
set(gca, 'YScale', 'log');

plot(SNRdBs, MseESPRIT,'r:s', 'LineWidth', 2);
plot(SNRdBs, MseMUSIC,'g:o', 'LineWidth', 2);
plot(SNRdBs, CRLB,'m:<', 'LineWidth', 2);

l = legend({  
        'ESPRIT',...    
        'MUSIC',...
        'CRLB'},...    
        'Interpreter','latex', 'Box','off'); 
l.FontSize = 12; 
