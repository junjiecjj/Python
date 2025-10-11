function cpm_sig = cpm_mod(inputs,h,sps,L,q,m)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 完成CPM调制
% input：输入码元
% h：调制指数
% sps：每个符号采样点数
% L：关联（记忆）长度
% q：脉冲积分函数（以向量形式给出，sps*L个点）
% m：调制进制数
% cpm_sig：CPM调制信号
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
info_length = length(inputs); %输入符号数目
if(L>1)
    inputs = [ones(1,L-1)*(1-m),inputs];  
    %这里是添加0时刻之前L-1个符号状态，为什么添1-m?
end

theta_n = zeros(1,info_length+1);             %n=0的初始相位状态
phases = zeros(1,sps*info_length);            %信号相位值
for n = 0:info_length-1
    %计算第n个符号对应的相位值也即[nT:(n+1)T]时间段
    In = n+L;%这是引用第n个符号的Inputs下标
    %t = [n*sps+1:(n+1)*sps];%第n个符号占据的时间
    if n>0
        %第n个符号对应的相位状态
        theta_n(n+1) = theta_n(n) + pi*h*inputs(In-L);  %In-L+1?
    end
    theta = zeros(1,sps);%关联状态
    for k = n-L+1:n
        %计算第n个符号的关联状态
        theta = theta + 2*pi*h*inputs(k+L)*q((n-k)*sps+1:(n+1-k)*sps);
    end
    
    %根据相位状态和关联状态计算相位值
    phases(n*sps+1:(n+1)*sps) = mod(theta_n(n+1) + theta,2*pi);       
end
cpm_sig = exp(1j*phases);%根据相位值计算复包络
