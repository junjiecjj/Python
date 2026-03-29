function pmf = ps64qam_mb_pmf(constellation, nu)
% 计算Maxwell-Boltzmann概率质量函数
% p(x) ∝ exp(-nu * |x|^2)

metric = exp(-nu * abs(constellation(:)).^2);
pmf = metric ./ sum(metric);
end
