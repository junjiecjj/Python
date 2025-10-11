%% 二分法求解方程
function result = binary_solution(scope,precision,func,parameter)
    % result:求解数据
    % target：目标数值
    % scope：目标初始区间[小值，大值]
    % precision： 目标数值精度
    % func：求解方程
    % parameter：相关参数,如alpha，target
    
    shape = size(parameter); 
    if shape(2) == 2 %仅有一个参数输入时
        while 1
            result = mean(scope); %求解区间均值
            target = func(result,parameter(1,1)); %预设Pfa
            difference = 1 / target - 1 / parameter(1,end); %求解与理想值差距
            if abs(difference) < precision || abs(scope(1,1) - scope(1,2)) < 0.001 %判断是否达到精度要求
                return;
            elseif difference < 0 %未达到精度要去继续二分
                scope(1,1) = result;
            else
                scope(1,2) = result;
            end
        end
    elseif shape(2)==3      %用于OS_CFAR门限因子的计算
        while 1
            result = mean(scope); %求解区间均值
            target = func(result,parameter(1,1),parameter(1,2)); %预设Pfa,输入parameter满足alpha,N,rate顺序
            difference = 1 / target- 1 / parameter(1,end); %求解与理想值差距
            if abs(difference) < precision || abs(scope(1,1) - scope(1,2)) < 0.001 %判断是否达到精度要求
                return;
            elseif difference<0 %未达到精度要去继续二分
                scope(1,1) = result;
            else
                scope(1,2) = result;
            end
        end
    elseif shape(2) == 4
        while 1
            result = mean(scope); %求解区间均值
            target = func(result,parameter(1,1),parameter(1,2),parameter(1,3)); %预设Pfa,输入parameter满足alpha,N,rate1,rate2顺序
            difference = 1 / target - 1 / parameter(1,end);
            if abs(difference) < precision || abs(scope(1,1) - scope(1,2)) < 0.001 %判断是否达到精度要求
                return;
            elseif difference < 0 %未达到精度要去继续二分
                scope(1,1)  = result;
            else
                scope(1,2) = result;
            end
        end
    end
end