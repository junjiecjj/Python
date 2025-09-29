function [ s_Sigma s_Delta ]=sigma_delta(s_A,s_B)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
s_Sigma=cell(1,length(s_A));
s_Delta=cell(1,length(s_A));
for m=1:length(s_A)
    s_Sigma{1,m}=s_A{1,m}+s_B{1,m};
    s_Delta{1,m}=s_A{1,m}-s_B{1,m};
end

