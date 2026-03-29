function Hx = ps64qam_entropy(pmf)
% 计算输入熵 H(X)
pmf = pmf(:);
Hx = -sum(pmf .* log2(max(pmf, realmin)));
end
