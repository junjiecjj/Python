%% CA-CFAR标称因子计算
function alpha = ca_threhold(Pfa,N)
    % Pfa:虚警概率
    % N：参考单元数
    alpha = N .* (Pfa .^ (-1 / N) - 1 );
end
