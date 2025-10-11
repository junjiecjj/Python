%% 理论检测阈值的计算:已知相关参数求解理论T
function T_opt = T_opt_solve(Pfa,lambda,dB,Range)
    %Pfa:虚警概率
    %lambda:指数分布参数
    %dB:背景功率
    %Range:距离单元范围
    %T_opt:理论检测阈值
    scope = [0,100]; %求解区间
    precision = 0.0001; %精度设置
    func = @cdf_exponential; %指数分布
    parameter = [lambda,Pfa];
    x = binary_solution_v1(scope,precision,func,parameter); %求解标称因子
    T_opt = ones(1,Range(1,end)) .* x;
    
    P = form_Power_DB2P(dB); %将功率db形式转为P
    for i = 2:length(Range)
        T_opt(1,Range(1,i-1) + 1:Range(1,i)) = T_opt(1,Range(1,i-1) + 1:Range(1,i)) .* P(1,i-1); 
    end
    T_opt(1,1) = x .* P(1,1); 
end