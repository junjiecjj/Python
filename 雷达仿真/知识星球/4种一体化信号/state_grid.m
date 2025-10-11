function [next_state,pre_state] = state_grid(L,m,h_p,h_m)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 获得状态转移网格，以当前状态及当前输入的下一状态表示
% L：记忆长度
% m：调制进制数
% h_p,h_m：调制指数参量
% next_state：形状为[state_num,m]，表示当前状态为state_index,当前输入为input_index时的下一个时刻的state_index
% pre_state：形状为[state_num,m], 表示当前状态为state_index，前一时刻的m种可能状态
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = h_m/h_p; %调制指数
if mod(h_m,2)==0
    phase_state_num = h_p;                  %可能的相位状态数
else
    phase_state_num = 2*h_p;
end
phase_gap = 2*pi/phase_state_num;%可能相位状态间的间隔
phase_states = (0:phase_state_num-1)*phase_gap;%可能的相位状态

state_num = phase_state_num*(m^(L-1)); %可能的状态数
inputs = (0:m-1)*2-m+1;   %可能的输入符号
current_input_num = m;    %当前输入符号可能数
next_state = zeros(state_num,current_input_num);
pre_state = zeros(state_num,current_input_num);

if(L==1)
    %当前状态为[θn],state_index = phase_index
    %下一状态为[θn+1],θn+1 = θn+π*h*In
    for state = 1:state_num
        phase_index = state;
        theta_n = phase_states(phase_index);
        for input = 1:current_input_num
            % 计算当前状态当前输入的下一状态，顺便也就计算出了下一状态的前一状态
            In = inputs(input);
            theta_n_plaus1 = mod(theta_n + pi*h*In,2*pi);%θn+1
            next_state(state,input) = theta_n_plaus1/phase_gap+1;
            for j = 1:m
                if(pre_state(theta_n_plaus1/phase_gap+1,j) ==0)
                   pre_state(theta_n_plaus1/phase_gap+1,j) = state;
                   %continue
                   break
                end
            end
        end
    end
elseif(L==2)
    % 状态为[θn,In_1],state_index = phase_index*m + In_1_index
    % 下一状态为[θn+1,In],θn+1 = θn+π*h*In_1
    for state = 1:state_num
        phase_index = floor((state-1)/m)+1;%θn下标
        theta_n = phase_states(phase_index);
        In_1_index = state-(phase_index-1)*m; %In-1下标
        In_1 = inputs(In_1_index);
        for input = 1:current_input_num
            %In = inputs(input);
            theta_n_plaus1 = mod(theta_n + pi*h*In_1,2*pi);%θn+1
            theta_n_plaus1_index = theta_n_plaus1/phase_gap+1;%θn+1的下标
            next_state(state,input) = int8((theta_n_plaus1_index-1)*m+input);
            for j = 1:m
                if(pre_state(next_state(state,input),j)==0)
                    pre_state(next_state(state,input),j)=state;
                    break
                end
            end
        end
    end
elseif(L==3)
    % 状态为[θn,In_1,In_2],state_index = phase_index*m^2 + In_1_index*m + In_2_index
    % 下一状态为[θn+1,In,In-1],θn+1 = θn+π*h*In_2
    for state = 1:state_num
        phase_index = floor((state-1)/m^2)+1;
        theta_n = phase_states(phase_index);
        In_1_index = floor((state-(phase_index-1)*m^2-1)/m)+1;
        In_2_index = state-(phase_index-1)*m^2-(In_1_index-1)*m;
        In_2 = inputs(In_2_index);
        for input = 1:current_input_num
            theta_n_plaus1 = mod(theta_n + pi*h*In_2,2*pi);%θn+1
            theta_n_plaus1_index = theta_n_plaus1/phase_gap+1;%θn+1的下标
            next_state(state,input) = (theta_n_plaus1_index-1)*m^2+(input-1)*m+In_1_index;
            for j = 1:m
                if(pre_state(next_state(state,input),j)==0)
                    pre_state(next_state(state,input),j)=state;
                    break
                end
            end
        end
    end
                
    
elseif(L==4)
    % 状态为[θn,In_1,In_2,In-3],state_index = phase_index*m^3 + In_1_index*m^2 + In_2_index*m + In_2_index
    % 下一状态为[θn+1,In,In-1,In_2],θn+1 = θn+π*h*In_3
    for state = 1:state_num
        phase_index = floor((state-1)/m^3)+1;
        theta_n = phase_states(phase_index);
        In_1_index = floor((state-(phase_index-1)*m^3-1)/m^2)+1;
        In_2_index = floor((state-(phase_index-1)*m^3-(In_1_index-1)*m^2-1)/m)+1;
        In_3_index = state-(phase_index-1)*m^3-(In_1_index-1)*m^2-(In_2_index-1)*m;
        In_3 = inputs(In_3_index);
        for input = 1:current_input_num
            theta_n_plaus1 = mod(theta_n + pi*h*In_3,2*pi);%θn+1
            theta_n_plaus1_index = theta_n_plaus1/phase_gap+1;%θn+1的下标
            next_state(state,input) = (theta_n_plaus1_index-1)*m^3+(input-1)*m^2+(In_1_index-1)*m+(In_2_index);
            for j = 1:m
                if(pre_state(next_state(state,input),j)==0)
                    pre_state(next_state(state,input),j)=state;
                    break
                end
            end
        end
    end
end
end
