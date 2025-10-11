function viterbi_symbol = viterbi_demod(noisy_sig,correlator,next_states,L,h_m,h_p,m,sps)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CPM的维特比译码解调算法
% noisy_sig：接收机接收信号
% correlator：计算各分支度量需要各状态本地复包络，形状为(state_num,m,:)表示当前状态为state_index,当前输入为input_index时的复包络
% pre_states：状态转移网格，形状为[state_num,m],表示当前状态为state_index,前一时刻状态m种可能的state_index
% next_states：状态转移网格，形状为[state_num,m]，表示当前状态为state_index,当前输入为input_index时的下一个时刻的state_index
% L：记忆长度
% h_m，h_p：调制系数参量
% m：调制阶数
% sps：sample per symbol
% viterbi_symbol：维特比译码解调算法解调得到的符号
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if mod(h_m,2)==0
    phase_state_num = h_p;                  %可能的相位状态数
else
    phase_state_num = 2*h_p;
end

state_num = phase_state_num*(m^(L-1)); %可能的状态数
input_num = m;
inputs = (0:m-1)*2-m+1;   %可能的输入符号

symbol_num = length(noisy_sig)/sps;

Infty = -1*1e10;
branch_metric = ones(state_num,m,3)*Infty;%每个时间段内的分支度量,计算前初始化为负无穷
%对于[nT,(n+1)T]时间段，可能的状态转移情况为[state_num,m]种可能
%(state_index,m,1)存的是状态转移的分支度量值，(state_index,m,2)存的状态转移对应的前一状态，
%(state_index,m,3)存的是状态转移对应的输入

path_metric = zeros(symbol_num,state_num,3);%全时间段内的路径度量
%(symbol_index,state_index,1)存的是到第symbol_index个符号的第state_index个状态的累计路径度量值
%(symbol_index,state_index,2)存的是到第symbol_index个符号的第state_index个状态的前一时刻状态，用于回溯
%(symbol_index,state_index,1)存的是到第symbol_index个符号的第state_index个状态的前一时刻输入

%% 首先初始化t = 0时刻为全0，计算t = [0:T]时间段的分支度量和路径度量，也即第一符号
y = noisy_sig(1:sps);%第一个符号的接收信号
for state = 1:state_num
    for input = 1:input_num
       In = inputs(input);%当前输入
       R = reshape(correlator(state,input,:),1,sps); %状态转移对应的本地复包络
       next_state = next_states(state,input);%下一状态
       metric_value = real(sum(y.*conj(R)));%与本地复包络做相关获得状态转移的度量值
       for j = 1:m
           if(branch_metric(next_state,j,1) == Infty)
               branch_metric(next_state,j,1)= metric_value;%状态转移分支度量值
               branch_metric(next_state,j,2) = state;%转态转移的前一状态
               branch_metric(next_state,j,3) = In;%状态转移对应的输入
               break
           end
       end
    end
end
temp = reshape(branch_metric(:,:,1),state_num,m);%提取分支度量值
[max_metric,index] = max(temp,[],2);%找到最大度量值及对应的下标
path_metric(1,:,1) = max_metric;%第一个symbol各状态的度量值
for i = 1:state_num
    path_metric(1,i,2) = branch_metric(i,index(i),2);%转移到第一个symbol幸存的前一个状态
    path_metric(1,i,3) = branch_metric(i,index(i),3);%转移到第一个symbol幸存的输入
end

%% 从前往后计算每个时间段的分支度量值以及累计路径度量值
for t = 2:symbol_num
    %计算第t个符号对应的分支度量即路径度量
    branch_metric = ones(state_num,m,3)*Infty;  %分支度量计算前初始化为负无穷
    y = noisy_sig((t-1)*sps+1:t*sps);           %第t个符号对应的输入
    for state = 1:state_num
        for input = 1:input_num
            In = inputs(input);%当前输入
            R = reshape(correlator(state,input,:),1,sps); %状态转移对应的本地复包络
            next_state = next_states(state,input);%下一状态
            metric_value = real(sum(y.*conj(R)));%与本地复包络做相关获得状态转移的分支度量值
            for j = 1:m
               if(branch_metric(next_state,j,1) == Infty)
                   branch_metric(next_state,j,1)= metric_value + path_metric(t-1,state,1);%当前转移分支度量值+前一状态累计路径度量值
                   branch_metric(next_state,j,2) = state;%转态转移的前一状态
                   branch_metric(next_state,j,3) = In;%状态转移对应的输入
                   break
               end
            end
        end
    end
    temp = reshape(branch_metric(:,:,1),state_num,m);%提取分支度量值
    [max_metric,index] = max(temp,[],2);%找到最大度量值及对应的下标
    path_metric(t,:,1) = max_metric;%第t个symbol各状态的度量值
    for i = 1:state_num
        path_metric(t,i,2) = branch_metric(i,index(i),2);%转移到第t个symbol幸存的前一个状态
        path_metric(t,i,3) = branch_metric(i,index(i),3);%转移到第t个symbol幸存的输入
    end
end

%% 然后根据路径度量值从后往前回溯
viterbi_symbol = zeros(1,symbol_num);
% 首先找到最佳路径的终点
temp = reshape(path_metric(symbol_num,:,1),1,state_num);%最终时刻各个状态的累计路径度量值
[~,index] = max(temp);%根据最终时刻各状态的累计路径度量值得到最终时刻的最佳状态
viterbi_symbol(symbol_num) = path_metric(symbol_num,index,3);%最终时刻最佳状态对应的输入最为最终时刻输入的估计值
pre_state = path_metric(symbol_num,index,2);%最终时刻最佳状态的前一状态用于回溯
%然后根据路径图从后往前回溯
for t = symbol_num-1:-1:1
    viterbi_symbol(t) = path_metric(t,pre_state,3); %根据状态得当前最佳输入
    pre_state = path_metric(t,pre_state,2);         %状态回溯         
end













