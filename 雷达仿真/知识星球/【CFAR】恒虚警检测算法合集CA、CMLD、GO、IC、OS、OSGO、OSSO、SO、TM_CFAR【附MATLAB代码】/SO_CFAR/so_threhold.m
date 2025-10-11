%% SO-CFAR标称因子计算
function alpha = so_threhold(Pfa,N)
    % Pfa:虚警概率
    % N：参考单元数
    scope = [0,100]; %区间
    precision = 0.01; %精度
    func = @SO_Pfa;
    parameter = [N,Pfa];
    alpha = binary_solution(scope,precision,func,parameter);
end
