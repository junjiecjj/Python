function [x,index] = SOMP(A,b,sparsity)
% OMP Algo to solve equation Ax = b
% Input:
% A: sensing matrix  Np*Nfft
% b: measured vector Np*J
% sparsity
% Output:
% x: reconstructed signal 
%Step 1
[~,J] = size(b);
index = []; k = 1; [Am, An] = size(A); r = b; x=zeros(An,J);
while k <= sparsity
    %Step 2
    cor = sum(abs(A'*r),2);
    [Rm,ind] = max(cor); 
    index = [index ind];
    %Step 3
    P = A(:,index)*inv(A(:,index)'*A(:,index))*A(:,index)';
    r = (eye(Am)-P)*b; 
    k=k+1;
end
%Step 5
xind = inv(A(:,index)'*A(:,index))*A(:,index)'*b;
x(index,:) = xind;
end