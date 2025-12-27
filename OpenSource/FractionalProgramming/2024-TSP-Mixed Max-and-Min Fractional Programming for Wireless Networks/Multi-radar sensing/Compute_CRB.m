function [CRB] = Compute_CRB(M,A,s,G,L,alpha,sigma,Nr)
CRB = zeros(M,1);
% compute CRB1
for m = 1:M
    v = kron(eye(L),A{m})*s{m};
    Q = sigma(m)*eye(Nr(m)*L);
    for i = 1:M
        if i~=m
            Q = (kron(eye(L),G{m,i})*s{i})*(kron(eye(L),G{m,i})*s{i})'+Q;
        end
    end
    CRB(m) = 1/(2*abs(alpha(m)^2))*(v'*inv(Q)*v)^(-1);
end
CRB = real(CRB);
end