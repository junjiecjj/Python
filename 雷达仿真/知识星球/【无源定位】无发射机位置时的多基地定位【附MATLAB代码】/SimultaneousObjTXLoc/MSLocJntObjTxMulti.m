function [theta]=MSLocJntObjTxMulti(RxPos,r,d,Q_r,Q_d,Q_s)
% [theta]=MSLocJntObjTxMulti(RxPos,ro,do,Q_r,Q_d,Q_s)
%
% This function realizes the two-stage closed-form algorithm for jointly
% estimating the unknown object and transmitter positions in the presence
% of receiver position errors using multiple transmitters.
%
% Input parameter list:
% RXPos:  (Dim x M), receiver position matrix, M is the number of receivers.         
% r:      (M*N x 1), indirect range measurements.
% d:      (M*N x 1), direct range measurements.
% Q_r:    (M*N x M*N), covariance matrix of indirect range measurements.
% Q_d:    (M*N x M*N), covariance matrix of direct range measurements.
% Q_s:    (Dim*M x Dim*M), covariance matrix of receiver position errors.
% 
% Output parameter list:
% theta:  estimated object(theta(1:Dim)) and transmitter(theta(Dim+1:2*Dim)) positions.
%
% The program can be used for 2D(Dim=2) or 3D(Dim=3) localization.
%
% Reference:
% Y. Zhang and K. C. Ho, "Multistatic localization in the absence
% of transmitter position," IEEE Trans. Signal Process., vol. 67, no. 18, 
% pp. 4745-4760, Sep. 2019.
% 
% Yang Zhang and K. C. Ho   12-20-2019
% 

[K,M]=size(RxPos);          % M=number of receivers
                            % K=dimension
N=length(d)/M;              % N=number of transmitters

RptCnt = 3;                 % number of repetitions in Stage-1 to recompute W1
Q=blkdiag(Q_r,Q_d);         % Covariance of measurements

%=========== construct related vector and matrix ============
diagro=[];
for i=1:N
    hr((i-1)*M+1:i*M)=0.5*(r((i-1)*M+1:i*M).^2-sum(RxPos.*RxPos)');
    hd((i-1)*M+1:i*M)=0.5*(d((i-1)*M+1:i*M).^2-sum(RxPos.*RxPos)');
    diagro=blkdiag(diagro,r((i-1)*M+1:i*M));
end
Gr=[-kron(ones(N,1),RxPos'),zeros(M*N,N*K),kron(eye(N),ones(M,1)),diagro,-0.5*kron(eye(N),ones(M,1))]';
Gd=[zeros(M*N,K),-kron(eye(N),RxPos'),zeros(M*N,2*N),0.5*kron(eye(N),ones(M,1))]';
h1=[hr';hd'];
G1=[Gr';Gd'];

%============= first stage =================================== 
B1=eye(2*M*N);
Dr=zeros(M*N,K*M);
Dd=zeros(M*N,K*M);
D1=[Dr;Dd];
W1=B1*Q*B1'+D1*Q_s*D1';
phi=inv(G1'*inv(W1)*G1)*G1'*inv(W1)*h1;

for k=1:RptCnt              % iterating stage-1 to improve the weighting matrix W1
    for i=1:M*N
        B1(i,i)=norm(phi(1:K)-RxPos(:,mod(i-1,M)+1));
        B1(i+M*N,i+M*N)=norm(phi(ceil(i/M)*K+1:(ceil(i/M)+1)*K)-RxPos(:,mod(i-1,M)+1));
        Dr(i,mod(i-1,M)*K+1:(mod(i-1,M)+1)*K)=(phi(1:K)-RxPos(:,mod(i-1,M)+1))';
        Dd(i,mod(i-1,M)*K+1:(mod(i-1,M)+1)*K)=(phi(ceil(i/M)*K+1:(ceil(i/M)+1)*K)-RxPos(:,mod(i-1,M)+1))';
    end
    D1=[Dr;Dd];
    W1=B1*Q*B1'+D1*Q_s*D1';
    phi=inv(G1'*inv(W1)*G1)*G1'*inv(W1)*h1;
end

%========== second stage =====================================
uEst=phi(1:K);
tEst=reshape(phi(K+1:(N+1)*K),K,N);
Delta=diag(phi((K+1)*(N+1):(N+1)*K+2*N));
T=[];
for j=1:N
    T=blkdiag(T,tEst(:,j)');
end
B2=[eye(K),zeros(K,K*N),zeros(K,N),zeros(K,N),zeros(K,N);
    zeros(K*N,K),eye(K*N),zeros(K*N,N),zeros(K*N,N),zeros(K*N,N);
    -T*(kron(ones(N,1),eye(K))),-kron(eye(N),uEst'),2*eye(N),zeros(N,N),zeros(N,N);
    -ones(N,1)*uEst',-T,2*eye(N),2*Delta,zeros(N,N);
    zeros(N,K),-T,zeros(N,N),zeros(N,N),eye(N)];
h2=[phi(1:K);phi(K+1:K*(N+1));2*phi(K*(N+1)+1:K*(N+1)+N);
    phi((K+1)*(N+1):K*(N+1)+2*N).^2+2*phi(K*(N+1)+1:K*(N+1)+N);phi(K*(N+1)+2*N+1:end)];

G2=[eye(K),zeros(K,K*N);
    zeros(K*N,K),eye(K*N);
    T*(kron(ones(N,1),eye(K))),kron(eye(N),uEst');
    ones(N,1)*uEst',T;
    zeros(N,K),T];

W2=B2*inv(G1'*inv(W1)*G1)*B2';
theta=inv(G2'*inv(W2)*G2)*G2'*inv(W2)*h2;

