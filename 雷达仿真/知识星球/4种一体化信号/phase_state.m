function correlator = phase_state(sps,m,h_m,h_p,q,L)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 储存计算附加增量需要的复包络值
% sps：sample per symbol
% m：调制的进制数
% h_m,h_p：调制指数参量
% q：脉冲函数，以向量形式给出，长度为sps*L
% L：记忆（关联）长度
% correlator：附加增量的互相关计算所需的本地复包络值
% 形状为(state_num,m,:)表示当前状态为state_index,当前输入为input_index时的复包络
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = h_m/h_p; %调制指数
if mod(h_m,2)==0
    phase_state_num = h_p;                  %可能的相位状态数
else
    phase_state_num = 2*h_p;
end
phase_gap = 2*pi/phase_state_num;%可能相位状态间的间隔
phase_states = (0:phase_state_num-1)*phase_gap;%可能的相位状态

% history_inputs = zeros(1,L-1); %L-1中可能的历史输入序列(In-1,In-2,In-L+1)
% for i = 1:m^(L-1)
%     %由于历史序列可能的情况太多（指数级别），故很难写出通式，还是采用具体L值情况讨论
% end

state_num = phase_state_num*(m^(L-1)); %可能的状态数
inputs = (0:m-1)*2-m+1;   %可能的输入符号
current_input_num = m;    %当前输入符号可能数
correlator = zeros(state_num,current_input_num,sps);%存的是每种情况下的一个符号时间段内的复包络值

if(L == 1)
    % 当前状态为[theta_n]
    for state = 1:state_num
        % state_index = phase_index
        phase_index = state;
        theta_n = phase_states(phase_index);%相位状态θn
        for input = 1:current_input_num
            %计算[state,input]对应的复包络
            In = inputs(input);%当前输入码元In
            theta = 2*pi*h*In*q;%关联状态
            phase = mod(theta_n + theta,2*pi);%根据相位状态及关联状态获得相位值
            correlator(state,input,:) = exp(1j*phase);
        end
    end
elseif(L==2)
    % 当前状态为[theta_n,In_1]
    for state = 1:state_num
        % state_index = phase_index*m + In_1_index
        phase_index = floor((state-1)/m)+1;%状态数对m取整得相位状态下标
        theta_n = phase_states(phase_index);
        In_1_index = mod(state-1,m) + 1;%状态数对m取余得In-1下标
        In_1 = inputs(In_1_index);%(n-1)T时刻输入
        for input = 1:current_input_num
            %计算[state,input]对应的复包络
            In = inputs(input);%当前输入码元In
            theta = 2*pi*h*In_1*q(sps+1:2*sps)+2*pi*h*In*q(1:sps);%关联状态
            phase = mod(theta_n + theta,2*pi);%根据相位状态及关联状态获得相位值
            correlator(state,input,:) = exp(1j*phase);
        end
    end    
elseif(L==3)
    % 当前状态为[theta_n,In_1,In_2]
    for state = 1:state_num
        % state_index = phase_index*m^2 + In_1_index*m + In_2_index
        phase_index = floor((state-1)/m^2)+1;
        theta_n = phase_states(phase_index);%当前相位状态
        In_1_index = floor((state-(phase_index-1)*m^2-1)/m)+1;
        In_1 = inputs(In_1_index);%(n-1)T时刻输入
        In_2_index = state-(phase_index-1)*m^2-(In_1_index-1)*m;
        In_2 = inputs(In_2_index);%(n-2)T时刻的输入
        for input = 1:current_input_num
            %计算[state,input]对应的复包络
            In = inputs(input);%nT时刻输入
            theta = 2*pi*h*(In_2*q(2*sps+1:3*sps) + In_1*q(sps+1:2*sps) + In*q(1:sps));%关联状态
            phase = mod(theta_n + theta,2*pi);%根据相位状态及关联状态获得相位值
            correlator(state,input,:) = exp(1j*phase);
        end
    end
elseif(L==4)
    % 当前状态为[theta_n,In_1,In_2，In-3]
    for state = 1:state_num
        % state_index = phase_index*m^3 + In_1_index*m^2 + In_2_index*m + In_2_index
        phase_index = floor((state-1)/m^3)+1;
        theta_n = phase_states(phase_index);%当前相位状态
        In_1_index = floor((state-(phase_index-1)*m^3-1)/m^2)+1;
        In_1 = inputs(In_1_index);%(n-1)T时刻输入
        In_2_index = floor((state-(phase_index-1)*m^3-(In_1_index-1)*m^2-1)/m)+1;
        In_2 = inputs(In_2_index);%(n-2)T时刻的输入
        In_3_index = state-(phase_index-1)*m^3-(In_1_index-1)*m^2-(In_2_index-1)*m;
        In_3 = inputs(In_3_index);%(n-3)T时刻的输入
        for input = 1:current_input_num
            %计算[state,input]对应的复包络
            In = inputs(input);%nT时刻输入
            theta = 2*pi*h*(In_3*q(3*sps+1:4*sps)+In_2*q(2*sps+1:3*sps)+In_1*q(sps+1:2*sps)+In*q(1:sps));
            phase = mod(theta_n + theta,2*pi);%根据相位状态及关联状态获得相位值
            correlator(state,input,:) = exp(1j*phase);
        end
    end
end
        
