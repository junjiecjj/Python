function CRLB=MSLocJntObjTxMultiCRLB_RxErr(RXPos,TXPos,ObjPos,Q_r,Q_d,Q_s)
% CRLB=MSLocJntObjTxMultiCRLB_RxErr(RXPos,TXPos,ObjPos,Q_r,Q_d,Q_s)
%
% This function computes the CRLB of the joint estimation of the unknown object
% and transmitter positions in the prsence of receiver position errors using multiple transmitters. 
%
% Input parameter list:
% RXPos   : (Dim x M), receiver position matrix, M is the number of receivers.         
% TXPos:    (Dim x N), transmitter position, N is the number of transmitters. 
% ObjPos:   (Dim x 1), object position.
% Q_r:      (M*N x M*N), covariance matrix of indirect range measurements.
% Q_d:      (M*N x M*N), covariance matrix of direct range measurements.
% Q_s:      (Dim*M x Dim*M), covariance matrix of receiver position errors.
% 
% Output parameter list:
% CRLB:     (Dim*(N+1) x Dim*(N+1)), CRLB matrix of object and transmitter
%           positions estimate.
%
% The program can be used for 2D(Dim=2) or 3D(Dim=3) localization.
%
% Reference:
% Y. Zhang and K. C. Ho, "Multistatic localization in the absence of 
% transmitter position," IEEE Trans. Signal Process.
% vol. 67, no. 18, pp. 4745-4760, 15 Sept.15, 2019.
% 
% Yang Zhang and K. C. Ho   09-2019
% 

[D,M]=size(RXPos);              % M=number of receivers
                                % D=dimension
[~,N]=size(TXPos);              % N=number of transmitters

rho_rs=[];
rho_ds=[];
rho_rtInd=[]; 
rho_rt=TXPos-repmat(ObjPos,1,N); rho_rt=rho_rt./(ones(D,1)*sqrt(sum(rho_rt.^2)));
for j=1:N
    rho_rtInd=blkdiag(rho_rtInd,rho_rt(:,j));
end
rho_rtAll=repmat(rho_rtInd,1,M);
for i=1:M       
    rho_u=(ObjPos-RXPos(:,i))/norm(ObjPos-RXPos(:,i))*ones(1,N);
    rho_ru(:,(i-1)*N+1:i*N)=rho_u-rho_rt;
    rho_dt=TXPos-repmat(RXPos(:,i),1,N); rho_dt=rho_dt./(ones(D,1)*sqrt(sum(rho_dt.^2)));
    rho_dtInd=[];
    rho_rs=blkdiag(rho_rs,-rho_u);
    rho_ds=blkdiag(rho_ds,-rho_dt);
    for j=1:N
        rho_dtInd=blkdiag(rho_dtInd,rho_dt(:,j));
    end
    rho_dtAll(:,(i-1)*N+1:i*N)=rho_dtInd;
end

rho_dut=[zeros(M*N,D) rho_dtAll'];
rho_rut=[rho_ru' rho_rtAll'];
rho_rs=rho_rs';
rho_ds=rho_ds';
X=rho_rut'*inv(Q_d)*rho_rut+rho_dut'*inv(Q_r)*rho_dut;
Y=rho_rut'*inv(Q_d)*rho_rs+rho_dut'*inv(Q_r)*rho_ds;
Z=rho_rs'*inv(Q_d)*rho_rs+rho_ds'*inv(Q_r)*rho_ds+inv(Q_s);
CRLB=inv(X-Y*inv(Z)*Y');

