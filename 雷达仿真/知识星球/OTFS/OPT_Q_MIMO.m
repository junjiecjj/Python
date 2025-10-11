function [Q,Q_sqrt,CAP] = OPT_Q_MIMO(Heff,Pt,No);
[U,S1,V] = svd(Heff);
Sigma = diag(S1);
t = length(Sigma);
for p = t:-1:1
    onebylam = (Pt + sum(No./Sigma(1:p).^2))/p;
    optP = onebylam - No./Sigma(1:p).^2;
    CAP = sum(log2(1+ (Sigma(1:p).^2 .* optP/No)));
    Q = V(:,1:p)*diag(optP)*(V(:,1:p))';
    Q_sqrt = V(:,1:p)*diag(sqrt(optP));
    if (optP(p) > 0)
        break
    end
end
