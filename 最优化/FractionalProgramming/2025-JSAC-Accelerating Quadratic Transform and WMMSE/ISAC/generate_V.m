function [v1,v2] = generate_V(M,P)
%UNTITLED5 此处显示有关此函数的摘要
%   此处显示详细说明
v1 = randn(M,1)+1i*randn(M,1); v2 = randn(M,1)+1i*randn(M,1);
v1 = v1/norm(v1)*sqrt(P(1)); v2 = v2/norm(v2)*sqrt(P(2));


end