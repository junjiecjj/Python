%% The Blahut Arimoto Algorithm for the Rate Distortion Function 
%% By Fuwang Dong 2024/05/3

function [Rate, Distortion] = SF_BA_Rate_Distortion(Source, Recover, Slope, p_S)

% Source    信源码本
% Recover   恢复码本
% Slope     率失真函数的斜率s
% p_S       信源的分布

%% Blahut Arimoto Algorithm

N_symbol              =  length(Source);
Distortion_matrix     =  (Recover-Source.').^2;    % 计算失真矩阵d(S,\hat{S}),以MSE为度量
N_curve               =  length(Slope);
Distortion            =  zeros(1,N_curve);   
Rate                  =  zeros(1,N_curve);
Iteration             =  20;                       %  BA算法的迭代次数

for s_iter  =  1:N_curve

    s                 =  Slope(s_iter);          
    Q_hatS_given_S    =  ones(N_symbol,N_symbol)/N_symbol;         % 初始化转移概率，列向量表示给定S的hat(S)的分布，即行数为hat(S),列数为S
    p_S_temp          =  repmat(p_S',N_symbol,1);                  % 将信源分布写成矩阵形式，以便与条件概率矩阵相乘

    for  iter =  1:Iteration

        t_hatS_opt      =  p_S.'*Q_hatS_given_S.';
        p_hatS          =  t_hatS_opt.';                           % 计算估计hat(S)的分布，为列向量

        for index_S     =  1:N_symbol
            % 更新条件概率矩阵
            Q_hatS_given_S(:,index_S)  =  p_hatS.*exp(s*Distortion_matrix(:,index_S))/(p_hatS'*exp(s*Distortion_matrix(:,index_S)));
        end
        
        p_S_hatS                     =  Q_hatS_given_S.*p_S_temp;                % p(s,hat(s))的联合概率分布矩阵
        Distortion_s                 =  sum(sum(p_S_hatS.*Distortion_matrix));   % 期望失真
        I_ShatS_s                    =  p_S_hatS.*log2(Q_hatS_given_S./p_hatS);
        I_ShatS_s(isnan(I_ShatS_s))  =  0;
        Rate_s                       =  sum(sum(I_ShatS_s));

    end

    % 存储斜率为s时的结果
    Distortion(s_iter)    =  Distortion_s;
    Rate(s_iter)          =  Rate_s;

end


end







