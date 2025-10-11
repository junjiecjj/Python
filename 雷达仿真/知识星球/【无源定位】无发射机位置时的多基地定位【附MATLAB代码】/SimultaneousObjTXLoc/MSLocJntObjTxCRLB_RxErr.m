function CRLB=MSLocJntObjTxCRLB_RxErr(RXPos,TXPos,ObjPos,Q_r,Q_d,Q_s)
% CRLB=MSLocJntObjTxCRLB_RxErr(RXPos,TXPos,ObjPos,Q_r,Q_d,Q_s)
%
% This function computes the CRLB of the joint estimation of the unknown object
% and transmitter positions in the presence of receiver position errors. 
%
% Input parameter list:
% RXPos   : (Dim  x M), receiver position matrix, M is the number of receivers.         
% TXPos:    (Dim  x 1), transmitter position.
% ObjPos:   (Dim  x 1), object position.
% Q_r:      (M x M), covariance matrix of indirect range measurements.
% Q_d:      (M x M), covariance matrix of direct range measurements.
% Q_s:      (Dim*M x Dim*M), covariance matrix of receiver position errors.
% 
% Output parameter list:
% CRLB:     (2Dim  x 2Dim ), CRLB matrix of object and transmitter
%            positions estimate.
%
% The program can be used for 2D(Dim =2) or 3D(Dim =3) localization.
%
% Reference:
% Y. Zhang and K. C. Ho, "Multistatic localization in the absence of 
% transmitter position," IEEE Trans. Signal Process., vol. 67, no. 18, 
% pp. 4745-4760, Sep. 2019.
% 
% Yang Zhang and K. C. Ho   12-20-2019
% 

[K,M]=size(RXPos);              % M=number of receivers
                                % K=dimension
rho_rs=[];
rho_ds=[];

rho_u=repmat(ObjPos,1,M)-RXPos; rho_u=rho_u./(ones(K,1)*sqrt(sum(rho_u.^2)));
rho_rt=(TXPos-ObjPos)/norm(ObjPos-TXPos)*ones(1,M);
rho_ru=rho_u-rho_rt;
rho_dt=repmat(TXPos,1,M)-RXPos; rho_dt=rho_dt./(ones(K,1)*sqrt(sum(rho_dt.^2)));

for i=1:M
    rho_rs=blkdiag(rho_rs,-rho_u(:,i));
    rho_ds=blkdiag(rho_ds,-rho_dt(:,i));
end
rho_rut=[rho_ru' rho_rt'];
rho_dut=[zeros(M,K) rho_dt'];
rho_rs=rho_rs';
rho_ds=rho_ds';
X=rho_rut'*inv(Q_d)*rho_rut+rho_dut'*inv(Q_r)*rho_dut;
Y=rho_rut'*inv(Q_d)*rho_rs+rho_dut'*inv(Q_r)*rho_ds;
Z=rho_rs'*inv(Q_d)*rho_rs+rho_ds'*inv(Q_r)*rho_ds+inv(Q_s);
CRLB=inv(X-Y*inv(Z)*Y');
end

