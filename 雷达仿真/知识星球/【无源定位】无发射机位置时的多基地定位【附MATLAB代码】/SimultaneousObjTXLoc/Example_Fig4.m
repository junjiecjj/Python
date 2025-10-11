% This program will reproduce Fig.4 in Y. Zhang and K. C. Ho, 
% "Multistatic Localization in the Absence of Transmitter Position,"
% IEEE Trans. Signal Process. vol. 67, no. 18, pp. 4745-4760, Sep. 2019.
%
% Yang Zhang and K. C. Ho   12-20-2019
%
%       Copyright (C) 2019
%       Computational Intelligence Signal Processing Laboratory
%       University of Missouri
%       Columbia, MO 65211, USA.
%       hod@missouri.edu
%

clc; clear all; warning('off');             % program initialization.
randn('state',0);                           % initialize random number generator.
L=2000;                                     % number of ensemble runs. 

uo=[2000 5000]';                            % true object position.

to=[0 0]';                                  % true single transmitter position 

x=[1000 1000 -1000 -1000];                  % true receiver position matrix.
y=[1000 -1000 1000 -1000];
so=[x; y];

[K,M]=size(so);                             % M=number of receivers
                                            % K=dimension

ro=sqrt(sum((repmat(uo,1,M)-so).^2))'+norm(uo-to);      % true indirect ranges
do=sqrt(sum((repmat(to,1,M)-so).^2))';                  % true direct ranges

snsePwr=0.1;                                            % receiver position noise power                                      
J=diag([5,5,40,40,20,20,10,10]); 
Q_s = J*snsePwr;                                        % Covariance matrix of receiver position noise. 

nsePwrAlldB=[-10:5:40];                                 % measurements noise power in log-scale
fprintf('Simulation in progress ... \n');
for nsePwrIdx=1:length(nsePwrAlldB)
    fprintf('%2d/%d:  10log10(nsePwr) = %2d \n',nsePwrIdx,length(nsePwrAlldB),nsePwrAlldB(nsePwrIdx));
    nsePwr=10^(nsePwrAlldB(nsePwrIdx)/10);              % measurement noise power 
    Q_r=eye(M)*nsePwr;                                  % Covariance matrix of indirect range noise
    Q_d=eye(M)*nsePwr;                                  % Covariance matrix of direct range noise  
    CRLB=MSLocJntObjTxCRLB_RxErr(so,to,uo,Q_r,Q_d,Q_s); % CRLB evaluation
    CRLBu(nsePwrIdx)=trace(CRLB(1:K,1:K));              % CRLB for the object position estimate.
    
    SimulationMSEu=0;
    for ii=1:L,
        r=ro+sqrt(nsePwr)*randn(M,1);                   % noisy indirect ranges
        d=do+sqrt(nsePwr)*randn(M,1);                   % noisy direct ranges  
        s=so+reshape(sqrtm(Q_s)*randn(M*K,1),K,M);      % noisy receiver positions  
        Theta=MSLocJntObjTx(s,r,d,Q_r,Q_d,Q_s);         % algebraic closed-form solution
        SimulationMSEu=SimulationMSEu+norm((Theta(1:K)-uo),2).^2;
    end
    MSEu(nsePwrIdx)=SimulationMSEu/L;
end
figure(4);
plot(nsePwrAlldB,10*log10(CRLBu),'-r');
hold  on;
plot(nsePwrAlldB,10*log10(MSEu),'dk','MarkerSize',8);
hold off;
grid on; xlabel('10 log10(\sigma^2(m^2))'); ylabel('10 log10(MSE(m^2))');
legend('CRLB','Proposed solution');

