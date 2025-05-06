function [W] = update_W_MUMIMO_ISAC(l,L_Com,DD,eta)
%UNTITLED7 此处显示有关此函数的摘要
%   此处显示详细说明
[Nt,K,Q,~] = size(L_Com);  W = zeros(Nt,K,Q); 
for q = 1:Q
    W(:,:,q) = (eta*eye(Nt)+DD(:,:,l))\L_Com(:,:,q,l);
end
end