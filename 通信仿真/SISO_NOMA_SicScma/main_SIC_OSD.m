%% successive interference cancellation(SIC)，BPSK
clc;clear;
warning off;
addpath('5G_polar_construction/');
addpath('Modulation');
addpath('Tools/');
addpath('Tools/Integer_partitions');
addpath('Decoders/');
addpath('Tools/OLD');
addpath('Channels/');
addpath('eBCH/');
%% 设置参数
max_runs = 1e12;    %最大仿真帧数
max_err = 300;      %单用户最大错误帧数
display_interval = 10;%显示间隔
vec = 0:2:20;%多用户SNR
%--------信噪比类型------
%SNR = 'Eb/N0';
SNR = 'SNR';
%SNR = 'Es/N0';
%----------信道类型--------
% channel = 'Rician_block_fading';
channel = 'Rician_quasi_static';%复杂度统计只适合准静态信道
% channel = 'AWGN';

K_ric = 0;%Rician信道因子线性;0-Rayleigh信道

%阶数
order = 6;
%OLDn_decoder分段数
D = 2;
%Fast_OSD参数
T_fastOSD = 3;%psc门限（校验子汉明重量）
lambda_fastOSD = 0.5;%pnc规则

%用户数
N_u = 2;
%相邻两用户功率倍数
ratio = 4;
%获得各用户功率
power = zeros(N_u,1);%各用户功率
rou = zeros(N_u,1);%各用户信号幅度
power(N_u) = 1;
for i_Nu = N_u-1:-1:1%用户功率递减
    power(i_Nu) = power(i_Nu+1)*ratio;
end
sum_power = sum(power);
for i_Nu = 1:N_u
    rou(i_Nu) = sqrt(power(i_Nu)/sum_power);
end



%译码器选择,只是将相应译码器名称写入文件
%Decoder = 'conventional_OSD_decoder';%指定order
Decoder = 'OperationNum_conventional_OSD_decoder';
%Decoder = 'trivial_discard_conventional_OSD_decoder';%指定order
%Decoder = 'fast_OSD_decoder';%指定order
%Decoder = 'OperationNum_fast_OSD_decoder';%指定order
%Decoder = 'OLD1_decoder';%无限制
%Decoder = 'max_test_OLD1_decoder';%指定最大TEPs数量
%Decoder = 'OLDn_decoder';%分D段，无限制
%Decoder = 'max_test_OLDn_decoder';%分D段，指定最大TEPs数量
%Decoder = 'OperationNum_OLDn_decoder';%分D段，无限制,统计操作数
%Decoder = 'old_OperationNum_OLD1_decoder';%分1段，无限制,统计操作数，原停止准则（LRB取min）
%Decoder = 'old_max_test_OLD1_decoder';%分1段，原停止准则（LRB取min）,限制最大TEP数量
%Decoder = 'old_OperationNum_OLDn_decoder';%分D段，无限制,统计操作数，原停止准则（LRB取min）
%Decoder = 'old_max_test_OLDn_decoder';%分D段，原停止准则（LRB取min）,限制最大TEP数量
%Decoder = 'PB_OSD_decoder';%指定order
%Decoder = 'OperationNum_PB_OSD_decoder';%指定order
%Decoder = 'SPB_OSD_decoder';%指定order

%% 编码方案
%****************************5G polar码****************************
% encoding_scheme = '5G polar';
% n = 64;    %实际码长
% k = 16;        %实际信息比特长度
% crc_length = 11; %crc校验位长度
% R = k/n;            %码率
% %crc generator matrix and parity check matrix
% poly = get_crc_poly(crc_length);
% [G_crc,~,~] = make_CRC_GH(k, poly);
% %5G Construction
% [N, rate_matching_mode, rate_matching_pattern, frozen_pattern] = Para_5GConstruction(k, E, crc_length);
% frozen_pattern_after_rate_matching = frozen_pattern(rate_matching_pattern);%速率兼容后的冻结模式
% %Polar生成矩阵
% G_polar_full = G_matrix(N);
% G_polar = G_polar_full(frozen_pattern==0,rate_matching_pattern);
% G_AA = G_polar_full(frozen_pattern==0,frozen_pattern==0);
% crc+polar生成矩阵
%----------高斯消元系统编码方法-------------
%G = mod(G_crc*G_polar,2);
%[G,perm_index] = my_Gauss_Elimination(G);

%****************************eBCH码****************************
encoding_scheme = 'eBCH';
% H = eBCH_n8_k4_dmin4_H();
% H = eBCH_n16_k7_dmin6_H();
% H = eBCH_n32_k11_dmin12_H();
H = eBCH_n64_k16_dmin24_H();
% H = eBCH_n64_k24_dmin16_H();
% H = eBCH_n64_k30_dmin14_H();
% H = eBCH_n64_k36_dmin12_H();
% H = eBCH_n128_k64_H();
n = size(H,2);
k = n-size(H,1);
R = k/n;    %码率
[H_sys,perm_index] = my_Gauss_Elimination(H);
G = [eye(k) H_sys(:,n-k+1:n)'];

%% 存储结果
bler = zeros(1,length(vec));
num_runs = zeros(1,length(vec));
ber = zeros(1,length(vec));
error_bits = zeros(1,length(vec));
error_blocks = zeros(1,length(vec));
total_test_num = zeros(1,length(vec));
test_num = zeros(1,length(vec));
ave_test_num = zeros(1,length(vec));
total_operation_num = zeros(1,length(vec));
operation_num = zeros(1,length(vec));
ave_operation_num = zeros(1,length(vec));
switch channel
    case 'Rician_block_fading'
        channel_use = n;
    case 'Rician_quasi_static'
        channel_use = 1;
    case 'AWGN'
        channel_use = 1;
end
h_ric = zeros(N_u,channel_use);%存储信道系数
info = zeros(N_u,k);%存储信息比特
%写入文件
name = ['.\Results\' encoding_scheme '_n' num2str(n) '_k' num2str(k)  '_' Decoder '_order' num2str(order) '.txt'];
filename = fopen(name,'a+');
fprintf(filename,'\n\n');
fprintf(filename,'程序开始时间：%s \n',datestr(now));
fprintf(filename,'n = %d  ',n);
fprintf(filename,'k = %d  ',k);fprintf(filename,'\n');
fprintf(filename,'max_runs = %1.2e  ',max_runs);
fprintf(filename,'max_err = %d  ',max_err);
fprintf(filename,'display_interval = %d  ',display_interval);fprintf(filename,'\n');
fprintf(filename,'encoding_scheme = %s  ', encoding_scheme);
fprintf(filename,'Decoder = %s  ', Decoder);fprintf(filename,'\n');
fprintf(filename,'order = %d  ', order);fprintf(filename,'\n');
fprintf(filename, 'SNR                BER               BLER           ave_test_num        ave_operation_num      total_blocks');fprintf(filename,'\n');

%% 仿真
%程序运行时间
tic

for i_vec = 1 : length(vec)
    %根据不同信噪比形式获得单路噪声功率
    switch SNR
        case 'Eb/N0'
           sigma = 1/sqrt(2*R*N_u) * 10^(-vec(i_vec)/20);%E_b/N_0，单路噪声功率
        case 'SNR'  
           sigma = 1/10^(vec(i_vec)/20);%SNR
           if contains(channel,'Rician')%若为复数信道，则sigma应为单路噪声功率
                sigma = sigma/sqrt(2);
           end
         case 'Es/N0'
           sigma = 1/sqrt(2) * 10^(-vec(i_vec)/20);%E_s/N_0，单路噪声功率 
    end
    
    for i_run = 1 : max_runs
        %叠加编码，Superposition Coding
        inteleaver_matrix = Interleaver(N_u,n);%生成交织序列
        y = zeros(1, n);%接收信号，初始化为全零序列
        %获得信道的系数
        for i_Nu = 1:N_u
            %------------瑞利/莱斯/AWGN信道-------------
            if contains(channel,'Rician')
                h_ric(i_Nu,:) = Ric_model(K_ric, channel_use);
            else
                h_ric(i_Nu,:) = 1;
            end
            %------------瑞利/莱斯/AWGN信道-------------
        end
%         if strcmp(channel,'Rician_quasi_static')
%             temp = abs(h_ric);%信道增益
%             [~, perm] = sort(temp,'ascend');%排序
%             h_ric(:,:) = h_ric(perm,:);%根据信道增益降序排列&&strcmp(equal_power,'yes')
%         end
%         if strcmp(channel,'Rician_quasi_static')
%             temp = 1/sum(1./abs(h_ric).^2);
%             rou = sqrt(temp./abs(h_ric).^2);
%         end
        h_ric = h_ric.*rou;%将信道系数和信号幅度看作整体
        %编码
        for i_Nu = 1:N_u
            info(i_Nu,:) = rand(1, k) > 0.5; %生成信源
            c = mod(info(i_Nu,:)*G,2);%编码
            symbol = 1-2*c;%BPSK调制
            symbol = symbol(inteleaver_matrix(i_Nu,:));%交织
            y_temp = h_ric(i_Nu,:).*symbol;
            y = y + y_temp;
        end
        %------------白噪声------------
        if contains(channel,'Rician')
            noise = randn(1, n) + randn(1, n) * 1j;
        else
            noise = randn(1, n);%AWGN,BPSK
        end
        y = y + sigma*noise;%加白噪声
        %------------白噪声------------
        
        %若为‘Rician_quasi_static’信道，且接收端已知信道系数，则可对信号最强的先译码
        if strcmp(channel,'Rician_quasi_static')
            temp = abs(h_ric);%信道增益
            [~, perm] = sort(temp,'descend');%排序
            h_ric(:,:) = h_ric(perm,:);%根据信道增益降序排列
            info(:,:) = info(perm,:);%相应的用户信息序列也置换
            inteleaver_matrix(:,:) = inteleaver_matrix(perm,:);%相应的交织序列也置换
            total_operation_num(i_vec) = total_operation_num(i_vec) + N_u*log(N_u)/log(2);%********各用户根据接收功率排序比较操作nlogn，
            total_operation_num(i_vec) = total_operation_num(i_vec) + 3*N_u;%********信道增益的复杂度，信道系数模的平方需要3个操作
        end
        
        
        for i_Nu = 1:N_u
            y_temp = y./h_ric(i_Nu,:);%对信噪比最高的信号均衡
            llr = 2/sigma^2*real(y_temp).*(abs(h_ric(i_Nu,:))).^2;%LLR
            total_operation_num(i_vec) = total_operation_num(i_vec) + n + n + n;%********均衡需要n个除法，计算LLR：第一个乘法需要n个，取实部不计入，信道系数模的平方需要3个操作，上述已计算，再相乘需要n个操作
            llr(inteleaver_matrix(i_Nu,:)) = llr;%解交织
            
            %***************************译码器选择*************************************
            %原始OSD
            %[test_num(i_vec), c_esti] = conventional_OSD_decoder(llr, G, order);
            %原始OSD,统计操作数
            [operation_num(i_vec),test_num(i_vec), c_esti] = OperationNum_conventional_OSD_decoder(llr, G, order);
            %原始OSD,加入trivial discard准则，速度较快，无性能损失
            %[test_num(i_vec), c_esti] = trivial_discard_conventional_OSD_decoder(llr, G, order);
            %fast-OSD
            %[test_num(i_vec), c_esti] = fast_OSD_decoder(llr, G, order, T_fastOSD, lambda_fastOSD);
            %fast-OSD,统计操作数
            %[operation_num(i_vec), test_num(i_vec), c_esti] = OperationNum_fast_OSD_decoder(llr, G, order, T_fastOSD, lambda_fastOSD, max_test_num);
            %proposed
            %[test_num(i_vec), c_esti] = OLD1_decoder(llr, G);
            %proposed,限制最大TEPs数量
            %[test_num(i_vec), c_esti] = max_test_OLD1_decoder(llr, G, max_test_num);
            %proposed，多段拟合
            %[test_num(i_vec), c_esti] = OLDn_decoder(llr, G, D);
            %proposed，多段拟合,限制最大TEPs数量
            %[test_num(i_vec), c_esti] = max_test_OLDn_decoder(llr, G, D, max_test_num);
            %proposed，多段拟合,统计操作数
            %[operation_num(i_vec), test_num(i_vec), c_esti] = OperationNum_OLDn_decoder(llr, G, D);
            %proposed，统计操作数，原停止准则（LRB取min）
            %[operation_num(i_vec), test_num(i_vec), c_esti] = old_OperationNum_OLD1_decoder(llr, G);
            %proposed，原停止准则（LRB取min），限制最大TEP数量
            %[test_num(i_vec), c_esti] = old_max_test_OLD1_decoder(llr, G, max_test_num);
            %proposed，多段拟合,统计操作数，原停止准则（LRB取min）
            %[operation_num(i_vec), test_num(i_vec), c_esti] = old_OperationNum_OLDn_decoder(llr, G, D);
            %proposed，多段拟合，原停止准则（LRB取min），限制最大TEP数量
            %[test_num(i_vec), c_esti] = old_max_test_OLDn_decoder(llr, G, D, max_test_num);
            %PB-OSD
            %[test_num(i_vec), c_esti] = PB_OSD_decoder(llr, sigma, G, order);
            %PB-OSD，统计操作数
            %[operation_num(i_vec), test_num(i_vec), c_esti] = OperationNum_PB_OSD_decoder(llr, sigma, G, order);
            %SPB-OSD
            %[test_num(i_vec), c_esti] = SPB_OSD_decoder(llr, sigma, G, order);
            %*************************************译码器选择*************************************
            %剔除已译出的信号
            temp = 1-2*c_esti;
            temp = temp(inteleaver_matrix(i_Nu,:));%交织
            y = y - h_ric(i_Nu,:).*temp;
            if i_Nu < N_u
                total_operation_num(i_vec) = total_operation_num(i_vec)+2*n;%**********调制操作
                total_operation_num(i_vec) = total_operation_num(i_vec)+2*n;%**********与信道系数相乘需要n个乘法，减去影响需要n个减法
            end
            info_esti = c_esti(1:k);%得到信息比特
            
            if any(info_esti ~= info(i_Nu,:))
                %误帧数加一
                error_blocks(i_vec) = error_blocks(i_vec) + 1;
                %误比特数
                error_bits(i_vec)=sum(mod((info_esti+info(i_Nu,:)), 2)) + error_bits(i_vec);
            end
            
            %总仿真帧数加一
            num_runs(i_vec) = num_runs(i_vec) + 1;
            %总操作数加
            total_operation_num(i_vec) = total_operation_num(i_vec) + operation_num(i_vec);
            %总测试TEPs次数加
            total_test_num(i_vec) = total_test_num(i_vec) + test_num(i_vec); 
            
        end
        %BER
        ber(i_vec) = error_bits(i_vec)/(num_runs(i_vec)*k);
        %BLER
        bler(i_vec) = error_blocks(i_vec)/num_runs(i_vec);
        %平均测试TEPs次数
        ave_test_num(i_vec) = total_test_num(i_vec)/num_runs(i_vec);
        %平均操作数
        ave_operation_num(i_vec) = total_operation_num(i_vec)/(num_runs(i_vec)*k);
        
        %若达到最大仿真帧数则跳出
        if error_blocks(i_vec) >= max_err*N_u
            break;
        end
        
        %每仿真display_interval帧，在命令行窗口输出一次
        if mod(i_run, display_interval) == 0
            disp(' ');            disp(['Sim iteration running = ' num2str(i_run)]);
            disp(['n = ' num2str(n) '  k = ' num2str(k) ] );
            disp([encoding_scheme '  ' Decoder]);
            disp(['order = ' num2str(order) ' D = ' num2str(D)]);
            disp( [SNR '         BER           BLER           ave_test_num    ave_operation_num    error_blocks']);
            disp(num2str([vec(1:i_vec)', ber(1:i_vec)', bler(1:i_vec)', ave_test_num(1:i_vec)', ave_operation_num(1:i_vec)', error_blocks(1:i_vec)']));
            disp(' ');
        end
    end
    fprintf(filename,'%f    %1.9f    %1.9f     %f     %f              %d\n',vec(i_vec), ber(i_vec), bler(i_vec), ave_test_num(i_vec), ave_operation_num(i_vec), num_runs(i_vec));
end
toc
disp(['程序结束时间：' datestr(now)]);
fprintf(filename,'程序运行时间：%s \n',num2str(toc));
fprintf(filename,'程序结束时间：%s \n',datestr(now));
fclose(filename);