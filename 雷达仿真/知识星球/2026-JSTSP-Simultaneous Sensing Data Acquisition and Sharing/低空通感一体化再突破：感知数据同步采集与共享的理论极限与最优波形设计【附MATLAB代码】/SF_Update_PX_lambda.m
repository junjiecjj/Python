function [P_x,lambda,cost] = SF_Update_PX_lambda(mu,X_input,gx,b,c,B)
    % lamda, u 为cost和失真的Lagrange乘子
    % xs, Px,  为信道输入及其概率分布
    % gx       为公式(21)的中间变量参数
    % b, c     为cost和失真函数
    % B        为cost的期望阈值
    %  内部循环，搜索满足cost约束的Lagrange乘子lambda
    lambda_lb                 =  -100;                                                          % 选取乘子的下界
    lambda_ub                 =  0;                                                             % 乘子的上界
    [~,cost_lb]               =  loop_cost(lambda_lb,mu,gx,X_input,b,c,B);     % 下界cost与阈值的差
    [~,cost_ub]               =  loop_cost(lambda_ub,mu,gx,X_input,b,c,B);     % 上界cost与阈值的差

    for   jj  =  1:30
        if ( cost_ub > 0) && (cost_lb < 0)
            lambda            =  (lambda_lb+lambda_ub)/2;
            [~,cost]          =  loop_cost(lambda,mu,gx,X_input,b,c,B);
            if  cost > 0
                lambda_ub     =  lambda;
                cost_ub       =  cost;
            else
                lambda_lb     =  lambda;
                cost_lb       =  cost;
            end
        end
    end
 [P_x,cost]                   =  loop_cost(lambda,mu,gx,X_input,b,c,B);
end

%%  计算给定Lambda时的期望cost函数，比较其与给定阈值之间的大小

function  [px,delta_cost]  =  loop_cost(lambda,mu,gx,X_input,b,c,B)
g            =  gx + lambda*b(X_input)-mu*c(X_input);   
% g            =  g - max(g);
px           =  exp(g) ./ sum(exp(g));                    % 根据式(20)求px
delta_cost   =  b(X_input).'*px-B;                        % 在此分布px下，求期望cost与给定阈值B的差异
end

