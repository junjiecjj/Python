function [W] = adjust_W_power(W,Pmax)
%UNTITLED8 此处显示有关此函数的摘要
%   此处显示详细说明 
[~,~,Q,L] = size(W); P = zeros(L,1);
for l = 1:L
    for q = 1:Q
        P(l) = P(l)+norm(W(:,:,q,l),'fro')^2;
    end
    for q = 1:Q
        W(:,:,q,l) = sqrt(Pmax/P(l))*W(:,:,q,l);
    end
end
end