function [W] = generate_W(Nt,K,Pmax,L,Q)
%UNTITLED8 此处显示有关此函数的摘要
%   此处显示详细说明
W = randn(Nt,K,Q,L)+1i*randn(Nt,K,Q,L); P = zeros(L,1);
for l = 1:L
    for q = 1:Q
        P(l) = P(l)+norm(W(:,:,q,l),'fro')^2;
    end
    for q = 1:Q
        W(:,:,q,l) = sqrt(Pmax/P(l))*W(:,:,q,l);
    end
end
end