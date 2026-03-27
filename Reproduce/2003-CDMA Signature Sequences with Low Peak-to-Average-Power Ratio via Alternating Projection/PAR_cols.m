


%% 辅助函数：计算各列 PAR
function par = PAR_cols(X)
    par = max(abs(X).^2, [], 1) ./ (mean(abs(X).^2, 1));
end