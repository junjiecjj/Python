% This program will reproduce Fig. 3 in Y. Zhang and K. C. Ho, 
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

to=[0 0]';                                  % true transmitter position

x=[1000 1000 -1000 -1000];                  % receiver position matrix.
y=[1000 -1000 1000 -1000];
s=[x; y];

[K,M]=size(s);                              % M=number of receivers
                                            % K=dimension

ro=sqrt(sum((repmat(uo,1,M)-s).^2))'+norm(uo-to);   % true indirect ranges
do=sqrt(sum((repmat(to,1,M)-s).^2))';               % true direct ranges

Qs=zeros(prod(size(s)));                      % no receiver position error

nsePwrAlldB=[-10:5:40];                       % noise power in log-scale

fprintf('Simulation in progress ... \n');
for nsePwrIdx=1:length(nsePwrAlldB) 
    fprintf('%2d/%d:  10log10(nsePwr) = %2d \n',nsePwrIdx,length(nsePwrAlldB),nsePwrAlldB(nsePwrIdx));
    nsePwr=10^(nsePwrAlldB(nsePwrIdx)/10);     % noise power 
    Q_r=eye(M)*nsePwr;                         % Covariance matrix of indirect range noise
    Q_d=eye(M)*nsePwr;                         % Covariance matrix of direct range noise
    CRLB=MSLocJntObjTxCRLB(s,to,uo,Q_r,Q_d);   % CRLB evaluation.
    CRLBmin(nsePwrIdx)=2.2046/M*nsePwr;        % minimum possible CRLB
    CRLBu(nsePwrIdx)=trace(CRLB(1:K,1:K));     % CRLB for the object position estimate.
    
    SimulationMSEu=0;
    for ii=1:L,
        r=ro+sqrt(nsePwr)*randn(M,1);               % noisy indirect ranges
        d=do+sqrt(nsePwr)*randn(M,1);               % noisy direct ranges
        theta=MSLocJntObjTx(s,r,d,Q_r,Q_d,Qs);      % algebraic closed-form solution
        SimulationMSEu=SimulationMSEu+norm((theta(1:K)-uo),2).^2;
    end  
    MSEu(nsePwrIdx)=SimulationMSEu/L;
end

figure(3);
plot(nsePwrAlldB,10*log10(CRLBmin),'-k','LineWidth',2);
hold  on;
plot(nsePwrAlldB,10*log10(CRLBu),'-r');
hold  on;
plot(nsePwrAlldB,10*log10(MSEu),'dk','MarkerSize',8);
hold off;
grid on; xlabel('10 log10(\sigma^2(m^2))'); ylabel('10 log10(MSE(m^2))');
legend('Minimum possible CRLB','CRLB for joint estimation','Proposed solution','location','northwest');
axis([-10 40 -20 82]);    
