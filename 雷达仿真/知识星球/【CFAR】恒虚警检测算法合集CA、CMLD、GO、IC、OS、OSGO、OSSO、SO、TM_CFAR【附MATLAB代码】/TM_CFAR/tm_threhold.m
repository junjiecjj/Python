%% TM-CFAR标称因子计算
function alpha = tm_threhold(Pfa,N,rate1,rate2)
    % Pfa:虚警概率
    % N：参考单元数
    scope = [0,100]; %区间
    precision = 0.01; %精度
    func = @TM_Pfa;
    parameter = [N,rate1,rate2,Pfa];
    alpha = binary_solution(scope,precision,func,parameter);
end