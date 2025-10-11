function [CRLB]=MSLocJntObjTxCRLB(RXPos,TXPos,ObjPos,Q_r,Q_d)
% [CRLB]=MSLocJntObjTxCRLB(RXPos,TXPos,ObjPos,Q_r,Q_d)
%
% This function computes the CRLB of the joint estimation of the unknown object
% and transmitter positions using both indirect- and direct-path range measurements. 
%
% Input parameter list:
% RXPos   : (Dim x M), receiver position matrix, M is the number of receivers.         
% TXPos:    (Dim x 1), transmitter position.
% ObjPos:   (Dim x 1), object position.
% Q_r:      (M x M), covariance matrix of indirect range measurements.
% Q_d:      (M x M), covariance matrix of direct range measurements.
% 
% Output parameter list:
% CRLB:     (2Dim x 2Dim), CRLB matrix of object and transmitter
%           position estimate.
%
% The program can be used for 2D(Dim=2) or 3D(Dim=3) localization.
%
% Reference:
% Y. Zhang and K. C. Ho, "Multistatic localization in the absence of 
% transmitter position," IEEE Trans. Signal Process., vol. 67, no. 18, 
% pp. 4745-4760, Sep. 2019.
% 
% Yang Zhang and K. C. Ho   12-20-2019
% 
%       Copyright (C) 2019
%       Computational Intelligence Signal Processing Laboratory
%       University of Missouri
%       Columbia, MO 65211, USA.
%       hod@missouri.edu
%

[K,M]=size(RXPos);    % M=number of receivers
                      % K=dimension

rho_u=repmat(ObjPos,1,M)-RXPos; rho_u=rho_u./(ones(K,1)*sqrt(sum(rho_u.^2)));
rho_rt=(TXPos-ObjPos)/norm(TXPos-ObjPos)*ones(1,M);
rho_ru=rho_u-rho_rt;
rho_dt=repmat(TXPos,1,M)-RXPos; rho_dt=rho_dt./(ones(K,1)*sqrt(sum(rho_dt.^2)));

rho_rtheta=[rho_ru' rho_rt'];
rho_dtheta=[zeros(M,K) rho_dt'];
FIM=rho_rtheta'*inv(Q_r)*rho_rtheta+rho_dtheta'*inv(Q_d)*rho_dtheta;
CRLB=inv(FIM);

