function [s_A s_B] = nose(s_A,s_B,k,B,F)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
for m=1:length(s_A)
a=size(s_A{1,m});
n_A=sqrt(k*B*F*290/2)*( randn(a)+j*randn(a));
n_B=sqrt(k*B*F*290/2)*( randn(a)+j*randn(a));
s_A{1,m}=s_A{1,m}+n_A;
s_B{1,m}=s_B{1,m}+n_B;

end

