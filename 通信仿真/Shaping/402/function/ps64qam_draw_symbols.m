function [symIdx, txSym] = ps64qam_draw_symbols(constellation, pmf, N)
% 按照给定PMF随机抽样发射符号
cdfVal = cumsum(pmf(:));
edges  = [0; cdfVal];
edges(end) = 1;   % 防止浮点误差导致最后一个区间丢失

u = rand(N,1);
symIdx = discretize(u, edges);

% 极少数边界点保护
symIdx(isnan(symIdx)) = length(pmf);

txSym = constellation(symIdx);
end
