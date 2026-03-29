function out = ps64qam_logsumexp_rows(A)
% 对矩阵A按行进行log-sum-exp运算
rowMax = max(A, [], 2);
out = rowMax + log(sum(exp(A - rowMax), 2));
end
