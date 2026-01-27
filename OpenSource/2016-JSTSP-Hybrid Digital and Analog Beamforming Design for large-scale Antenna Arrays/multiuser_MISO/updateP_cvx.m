


function [P, curpow, flag] = updateP_cvx(Qt, beta, Ps, K, sigma2)
    % updateP 函数 - 将Python的CVXPY代码转换为MATLAB的CVX
    
    lamba = 1;  % 这个变量在原始代码中未使用
    qkk = real(diag(Qt));
    
    % 使用CVX解决凸优化问题
    cvx_begin quiet
        variable x(K) nonnegative
        maximize(sum(beta .* log(1 + x/sigma2)))
        subject to
            sum(x .* qkk) == Ps
    cvx_end
    
    if strcmp(cvx_status, 'Solved')
        % 创建对角矩阵P
        P = diag(x);
        
        % 计算容量
        Cap = sum(beta .* log2(1 + diag(P)/sigma2));
        
        % 计算当前功率
        curpow = sum(x .* qkk);
        
        % 输出结果
        fprintf('      updateP, %s, %.4f/%.4f, %s, %.4f/%.4f\n', ...
            cvx_status, curpow, Ps, mat2str(x', 4), ...
            sum(beta .* log(1 + x/sigma2))*log2(exp(1)), Cap);
        
        flag = 1;
    else
        % 如果求解失败
        P = eye(K);
        curpow = 0;
        flag = 0;
        fprintf('      updateP failed with status: %s\n', cvx_status);
    end
end