function [theta]=MSLocJntObjTx(RxPos,r,d,Q_r,Q_d,Q_s)
% [theta]=MSLocJntObjTx(RxPos,r,d,Q_r,Q_d,Q_s)
%
% This function realizes the algebraic closed-form algorithm for jointly
% estimating the unknown object and transmitter positions in the presence
% of receiver position errors using both indirect- and direct-path range
% measurements.
%
% Input parameter list:
% RXPos:  (Dim x M), receiver position matrix, M is the number of receivers.         
% r:      (M x 1), indirect-path range measurements.
% d:      (M x 1), direct-path range measurements.
% Q_r:    (M x M), covariance matrix of indirect range measurements.
% Q_d:    (M x M), covariance matrix of direct range measurements.
% Q_s:    (Dim*M x Dim*M), covariance matrix of receiver position errors.
% 
% Output parameter list:
% theta:  Estimated object(theta(1:Dim)) and transmitter(theta(Dim+1:2*Dim)) positions.
%
% The program can be used for 2D(Dim=2) or 3D(Dim=3) localization.
%
% Reference:
% Y. Zhang and K. C. Ho, "Multistatic localization in the absence
% of transmitter position," IEEE Trans. Signal Process.
% vol. 67, no. 18, pp. 4745-4760, 15 Sept.15, 2019.
% 
% Yang Zhang and K. C. Ho   12-20-2019
%
%       Copyright (C) 2019
%       Computational Intelligence Signal Processing Laboratory
%       University of Missouri
%       Columbia, MO 65211, USA.
%       hod@missouri.edu
%

[K,M]=size(RxPos);            % M=number of receivers
                              % K=dimension
RptCnt = 3;                   % number of repetitions in Stage-1 to recompute W1
Q=blkdiag(Q_r,Q_d);           % Covariance of measurements

%=========== construct related vector and matrix ============
for i=1:M
    hr(i,:)=0.5*(r(i)^2-RxPos(:,i)'*RxPos(:,i));
    hd(i,:)=0.5*(d(i)^2-RxPos(:,i)'*RxPos(:,i));
    Gr(i,:)=[-RxPos(:,i)' zeros(1,K) 1 r(i)  -0.5]; 
    Gd(i,:)=[zeros(1,K) -RxPos(:,i)' 0 0  0.5]; 
end
h1=[hr;hd];
G1=[Gr;Gd];

%============= first stage ===================================  
Dr=zeros(M,K*M);
Dd=zeros(M,K*M);
B1=eye(2*M);
D1=[Dr;Dd];
W1=B1*Q*B1'+D1*Q_s*D1';
phi=inv(G1'*inv(W1)*G1)*G1'*inv(W1)*h1;

for k=1:RptCnt                              % iterating stage1 to improve the weighting matrix W1
    for i=1:M
        B1(i,i)=norm(phi(1:K)-RxPos(:,i));
        B1(i+M,i+M)=norm(phi(K+1:2*K)-RxPos(:,i));
        Dr(i,2*i-1:2*i)=(phi(1:K)-RxPos(:,i))';
        Dd(i,2*i-1:2*i)=(phi(K+1:2*K)-RxPos(:,i))';
    end
    D1=[Dr;Dd];
    W1=B1*Q*B1'+D1*Q_s*D1';
    phi=inv(G1'*inv(W1)*G1)*G1'*inv(W1)*h1;
end

%========== second stage =====================================
B2=[eye(K),zeros(K,K),zeros(K,1),zeros(K,1),zeros(K,1);
    zeros(K,K),eye(K),zeros(K,1),zeros(K,1),zeros(K,1);
    -phi(K+1:2*K)',-phi(1:K)',2,0,0;
    -phi(1:K)',-phi(K+1:2*K)',2,2*phi(2*K+2),0;
    zeros(1,K),-phi(K+1:2*K)',0,0,1];
h2=[phi(1:2*K);2*phi(2*K+1);2*phi(2*K+1)+phi(2*K+2)^2;phi(2*K+3)];
G2=[eye(K),zeros(K,K);
    zeros(K,K),eye(K);
    phi(K+1:2*K)',phi(1:K)';
    phi(1:K)',phi(K+1:2*K)';
    zeros(1,K),phi(K+1:2*K)'];
W2=B2*inv(G1'*inv(W1)*G1)*B2';
theta=inv(G2'*inv(W2)*G2)*G2'*inv(W2)*h2;

