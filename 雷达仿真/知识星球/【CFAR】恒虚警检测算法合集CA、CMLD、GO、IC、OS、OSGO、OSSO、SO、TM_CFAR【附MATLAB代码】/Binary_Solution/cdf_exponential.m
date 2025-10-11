%%指数分布的累积分布函数
function fc = cdf_exponential(x,lambda)
%      x:杂波幅度
% lambda:杂波参数
fc = 1 - exp(-x ./ lambda);
end