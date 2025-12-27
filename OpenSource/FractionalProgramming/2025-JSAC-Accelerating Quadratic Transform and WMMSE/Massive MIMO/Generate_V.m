function [V] = Generate_V(M,Q,L,Pmax)
% Randomly initialize V
V = randn(M,Q,L)+1i*randn(M,Q,L);
for l = 1:L
    sum_norm = 0;
    for q = 1:Q
        sum_norm = norm(V(:,q,l))^2+sum_norm;
    end
    V(:,:,l) = V(:,:,l)*sqrt(Pmax/sum_norm);
end
end