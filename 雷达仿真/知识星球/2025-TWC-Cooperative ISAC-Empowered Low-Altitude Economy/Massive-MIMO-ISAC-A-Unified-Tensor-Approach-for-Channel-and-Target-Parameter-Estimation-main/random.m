function r = random(name, A, B, varargin)
% RANDOM 生成随机数的替代函数（伪装成工具箱函数）
% 它可以骗过主程序，让其以为安装了工具箱

    % 处理维度参数
    if nargin < 4
        sz = [1, 1];
    else
        % 获取输入的维度，例如 random(..., 1, L)
        sz = [varargin{:}]; 
    end

    switch lower(name)
        case 'unif'
            % 均匀分布：对应 random('unif', Lower, Upper, m, n)
            % 算法：Lower + (Upper - Lower) * rand
            r = A + (B - A) .* rand(sz);
            
        case 'norm'
            % 正态分布：对应 random('norm', mu, sigma, m, n)
            % 算法：mu + sigma * randn
            r = A + B .* randn(sz);
            
        case 'rayl'
            % 瑞利分布：对应 random('rayl', B, m, n)
            % 这里 A 位置对应参数 B
            r = sqrt( (-2 * A^2) * log(1 - rand(sz)) );
            
        otherwise
            error('在这个简易替代版 random 函数中，暂不支持 %s 分布。建议直接使用 rand 或 randn。', name);
    end
end