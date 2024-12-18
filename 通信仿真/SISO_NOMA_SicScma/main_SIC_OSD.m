%% successive interference cancellation(SIC)��BPSK
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
%% ���ò���
max_runs = 1e12;    %������֡��
max_err = 300;      %���û�������֡��
display_interval = 10;%��ʾ���
vec = 0:2:20;%���û�SNR
%--------���������------
%SNR = 'Eb/N0';
SNR = 'SNR';
%SNR = 'Es/N0';
%----------�ŵ�����--------
% channel = 'Rician_block_fading';
channel = 'Rician_quasi_static';%���Ӷ�ͳ��ֻ�ʺ�׼��̬�ŵ�
% channel = 'AWGN';

K_ric = 0;%Rician�ŵ���������;0-Rayleigh�ŵ�

%����
order = 6;
%OLDn_decoder�ֶ���
D = 2;
%Fast_OSD����
T_fastOSD = 3;%psc���ޣ�У���Ӻ���������
lambda_fastOSD = 0.5;%pnc����

%�û���
N_u = 2;
%�������û����ʱ���
ratio = 4;
%��ø��û�����
power = zeros(N_u,1);%���û�����
rou = zeros(N_u,1);%���û��źŷ���
power(N_u) = 1;
for i_Nu = N_u-1:-1:1%�û����ʵݼ�
    power(i_Nu) = power(i_Nu+1)*ratio;
end
sum_power = sum(power);
for i_Nu = 1:N_u
    rou(i_Nu) = sqrt(power(i_Nu)/sum_power);
end



%������ѡ��,ֻ�ǽ���Ӧ����������д���ļ�
%Decoder = 'conventional_OSD_decoder';%ָ��order
Decoder = 'OperationNum_conventional_OSD_decoder';
%Decoder = 'trivial_discard_conventional_OSD_decoder';%ָ��order
%Decoder = 'fast_OSD_decoder';%ָ��order
%Decoder = 'OperationNum_fast_OSD_decoder';%ָ��order
%Decoder = 'OLD1_decoder';%������
%Decoder = 'max_test_OLD1_decoder';%ָ�����TEPs����
%Decoder = 'OLDn_decoder';%��D�Σ�������
%Decoder = 'max_test_OLDn_decoder';%��D�Σ�ָ�����TEPs����
%Decoder = 'OperationNum_OLDn_decoder';%��D�Σ�������,ͳ�Ʋ�����
%Decoder = 'old_OperationNum_OLD1_decoder';%��1�Σ�������,ͳ�Ʋ�������ԭֹͣ׼��LRBȡmin��
%Decoder = 'old_max_test_OLD1_decoder';%��1�Σ�ԭֹͣ׼��LRBȡmin��,�������TEP����
%Decoder = 'old_OperationNum_OLDn_decoder';%��D�Σ�������,ͳ�Ʋ�������ԭֹͣ׼��LRBȡmin��
%Decoder = 'old_max_test_OLDn_decoder';%��D�Σ�ԭֹͣ׼��LRBȡmin��,�������TEP����
%Decoder = 'PB_OSD_decoder';%ָ��order
%Decoder = 'OperationNum_PB_OSD_decoder';%ָ��order
%Decoder = 'SPB_OSD_decoder';%ָ��order

%% ���뷽��
%****************************5G polar��****************************
% encoding_scheme = '5G polar';
% n = 64;    %ʵ���볤
% k = 16;        %ʵ����Ϣ���س���
% crc_length = 11; %crcУ��λ����
% R = k/n;            %����
% %crc generator matrix and parity check matrix
% poly = get_crc_poly(crc_length);
% [G_crc,~,~] = make_CRC_GH(k, poly);
% %5G Construction
% [N, rate_matching_mode, rate_matching_pattern, frozen_pattern] = Para_5GConstruction(k, E, crc_length);
% frozen_pattern_after_rate_matching = frozen_pattern(rate_matching_pattern);%���ʼ��ݺ�Ķ���ģʽ
% %Polar���ɾ���
% G_polar_full = G_matrix(N);
% G_polar = G_polar_full(frozen_pattern==0,rate_matching_pattern);
% G_AA = G_polar_full(frozen_pattern==0,frozen_pattern==0);
% crc+polar���ɾ���
%----------��˹��Ԫϵͳ���뷽��-------------
%G = mod(G_crc*G_polar,2);
%[G,perm_index] = my_Gauss_Elimination(G);

%****************************eBCH��****************************
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
R = k/n;    %����
[H_sys,perm_index] = my_Gauss_Elimination(H);
G = [eye(k) H_sys(:,n-k+1:n)'];

%% �洢���
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
h_ric = zeros(N_u,channel_use);%�洢�ŵ�ϵ��
info = zeros(N_u,k);%�洢��Ϣ����
%д���ļ�
name = ['.\Results\' encoding_scheme '_n' num2str(n) '_k' num2str(k)  '_' Decoder '_order' num2str(order) '.txt'];
filename = fopen(name,'a+');
fprintf(filename,'\n\n');
fprintf(filename,'����ʼʱ�䣺%s \n',datestr(now));
fprintf(filename,'n = %d  ',n);
fprintf(filename,'k = %d  ',k);fprintf(filename,'\n');
fprintf(filename,'max_runs = %1.2e  ',max_runs);
fprintf(filename,'max_err = %d  ',max_err);
fprintf(filename,'display_interval = %d  ',display_interval);fprintf(filename,'\n');
fprintf(filename,'encoding_scheme = %s  ', encoding_scheme);
fprintf(filename,'Decoder = %s  ', Decoder);fprintf(filename,'\n');
fprintf(filename,'order = %d  ', order);fprintf(filename,'\n');
fprintf(filename, 'SNR                BER               BLER           ave_test_num        ave_operation_num      total_blocks');fprintf(filename,'\n');

%% ����
%��������ʱ��
tic

for i_vec = 1 : length(vec)
    %���ݲ�ͬ�������ʽ��õ�·��������
    switch SNR
        case 'Eb/N0'
           sigma = 1/sqrt(2*R*N_u) * 10^(-vec(i_vec)/20);%E_b/N_0����·��������
        case 'SNR'  
           sigma = 1/10^(vec(i_vec)/20);%SNR
           if contains(channel,'Rician')%��Ϊ�����ŵ�����sigmaӦΪ��·��������
                sigma = sigma/sqrt(2);
           end
         case 'Es/N0'
           sigma = 1/sqrt(2) * 10^(-vec(i_vec)/20);%E_s/N_0����·�������� 
    end
    
    for i_run = 1 : max_runs
        %���ӱ��룬Superposition Coding
        inteleaver_matrix = Interleaver(N_u,n);%���ɽ�֯����
        y = zeros(1, n);%�����źţ���ʼ��Ϊȫ������
        %����ŵ���ϵ��
        for i_Nu = 1:N_u
            %------------����/��˹/AWGN�ŵ�-------------
            if contains(channel,'Rician')
                h_ric(i_Nu,:) = Ric_model(K_ric, channel_use);
            else
                h_ric(i_Nu,:) = 1;
            end
            %------------����/��˹/AWGN�ŵ�-------------
        end
%         if strcmp(channel,'Rician_quasi_static')
%             temp = abs(h_ric);%�ŵ�����
%             [~, perm] = sort(temp,'ascend');%����
%             h_ric(:,:) = h_ric(perm,:);%�����ŵ����潵������&&strcmp(equal_power,'yes')
%         end
%         if strcmp(channel,'Rician_quasi_static')
%             temp = 1/sum(1./abs(h_ric).^2);
%             rou = sqrt(temp./abs(h_ric).^2);
%         end
        h_ric = h_ric.*rou;%���ŵ�ϵ�����źŷ��ȿ�������
        %����
        for i_Nu = 1:N_u
            info(i_Nu,:) = rand(1, k) > 0.5; %������Դ
            c = mod(info(i_Nu,:)*G,2);%����
            symbol = 1-2*c;%BPSK����
            symbol = symbol(inteleaver_matrix(i_Nu,:));%��֯
            y_temp = h_ric(i_Nu,:).*symbol;
            y = y + y_temp;
        end
        %------------������------------
        if contains(channel,'Rician')
            noise = randn(1, n) + randn(1, n) * 1j;
        else
            noise = randn(1, n);%AWGN,BPSK
        end
        y = y + sigma*noise;%�Ӱ�����
        %------------������------------
        
        %��Ϊ��Rician_quasi_static���ŵ����ҽ��ն���֪�ŵ�ϵ������ɶ��ź���ǿ��������
        if strcmp(channel,'Rician_quasi_static')
            temp = abs(h_ric);%�ŵ�����
            [~, perm] = sort(temp,'descend');%����
            h_ric(:,:) = h_ric(perm,:);%�����ŵ����潵������
            info(:,:) = info(perm,:);%��Ӧ���û���Ϣ����Ҳ�û�
            inteleaver_matrix(:,:) = inteleaver_matrix(perm,:);%��Ӧ�Ľ�֯����Ҳ�û�
            total_operation_num(i_vec) = total_operation_num(i_vec) + N_u*log(N_u)/log(2);%********���û����ݽ��չ�������Ƚϲ���nlogn��
            total_operation_num(i_vec) = total_operation_num(i_vec) + 3*N_u;%********�ŵ�����ĸ��Ӷȣ��ŵ�ϵ��ģ��ƽ����Ҫ3������
        end
        
        
        for i_Nu = 1:N_u
            y_temp = y./h_ric(i_Nu,:);%���������ߵ��źž���
            llr = 2/sigma^2*real(y_temp).*(abs(h_ric(i_Nu,:))).^2;%LLR
            total_operation_num(i_vec) = total_operation_num(i_vec) + n + n + n;%********������Ҫn������������LLR����һ���˷���Ҫn����ȡʵ�������룬�ŵ�ϵ��ģ��ƽ����Ҫ3�������������Ѽ��㣬�������Ҫn������
            llr(inteleaver_matrix(i_Nu,:)) = llr;%�⽻֯
            
            %***************************������ѡ��*************************************
            %ԭʼOSD
            %[test_num(i_vec), c_esti] = conventional_OSD_decoder(llr, G, order);
            %ԭʼOSD,ͳ�Ʋ�����
            [operation_num(i_vec),test_num(i_vec), c_esti] = OperationNum_conventional_OSD_decoder(llr, G, order);
            %ԭʼOSD,����trivial discard׼���ٶȽϿ죬��������ʧ
            %[test_num(i_vec), c_esti] = trivial_discard_conventional_OSD_decoder(llr, G, order);
            %fast-OSD
            %[test_num(i_vec), c_esti] = fast_OSD_decoder(llr, G, order, T_fastOSD, lambda_fastOSD);
            %fast-OSD,ͳ�Ʋ�����
            %[operation_num(i_vec), test_num(i_vec), c_esti] = OperationNum_fast_OSD_decoder(llr, G, order, T_fastOSD, lambda_fastOSD, max_test_num);
            %proposed
            %[test_num(i_vec), c_esti] = OLD1_decoder(llr, G);
            %proposed,�������TEPs����
            %[test_num(i_vec), c_esti] = max_test_OLD1_decoder(llr, G, max_test_num);
            %proposed��������
            %[test_num(i_vec), c_esti] = OLDn_decoder(llr, G, D);
            %proposed��������,�������TEPs����
            %[test_num(i_vec), c_esti] = max_test_OLDn_decoder(llr, G, D, max_test_num);
            %proposed��������,ͳ�Ʋ�����
            %[operation_num(i_vec), test_num(i_vec), c_esti] = OperationNum_OLDn_decoder(llr, G, D);
            %proposed��ͳ�Ʋ�������ԭֹͣ׼��LRBȡmin��
            %[operation_num(i_vec), test_num(i_vec), c_esti] = old_OperationNum_OLD1_decoder(llr, G);
            %proposed��ԭֹͣ׼��LRBȡmin�����������TEP����
            %[test_num(i_vec), c_esti] = old_max_test_OLD1_decoder(llr, G, max_test_num);
            %proposed��������,ͳ�Ʋ�������ԭֹͣ׼��LRBȡmin��
            %[operation_num(i_vec), test_num(i_vec), c_esti] = old_OperationNum_OLDn_decoder(llr, G, D);
            %proposed�������ϣ�ԭֹͣ׼��LRBȡmin�����������TEP����
            %[test_num(i_vec), c_esti] = old_max_test_OLDn_decoder(llr, G, D, max_test_num);
            %PB-OSD
            %[test_num(i_vec), c_esti] = PB_OSD_decoder(llr, sigma, G, order);
            %PB-OSD��ͳ�Ʋ�����
            %[operation_num(i_vec), test_num(i_vec), c_esti] = OperationNum_PB_OSD_decoder(llr, sigma, G, order);
            %SPB-OSD
            %[test_num(i_vec), c_esti] = SPB_OSD_decoder(llr, sigma, G, order);
            %*************************************������ѡ��*************************************
            %�޳���������ź�
            temp = 1-2*c_esti;
            temp = temp(inteleaver_matrix(i_Nu,:));%��֯
            y = y - h_ric(i_Nu,:).*temp;
            if i_Nu < N_u
                total_operation_num(i_vec) = total_operation_num(i_vec)+2*n;%**********���Ʋ���
                total_operation_num(i_vec) = total_operation_num(i_vec)+2*n;%**********���ŵ�ϵ�������Ҫn���˷�����ȥӰ����Ҫn������
            end
            info_esti = c_esti(1:k);%�õ���Ϣ����
            
            if any(info_esti ~= info(i_Nu,:))
                %��֡����һ
                error_blocks(i_vec) = error_blocks(i_vec) + 1;
                %�������
                error_bits(i_vec)=sum(mod((info_esti+info(i_Nu,:)), 2)) + error_bits(i_vec);
            end
            
            %�ܷ���֡����һ
            num_runs(i_vec) = num_runs(i_vec) + 1;
            %�ܲ�������
            total_operation_num(i_vec) = total_operation_num(i_vec) + operation_num(i_vec);
            %�ܲ���TEPs������
            total_test_num(i_vec) = total_test_num(i_vec) + test_num(i_vec); 
            
        end
        %BER
        ber(i_vec) = error_bits(i_vec)/(num_runs(i_vec)*k);
        %BLER
        bler(i_vec) = error_blocks(i_vec)/num_runs(i_vec);
        %ƽ������TEPs����
        ave_test_num(i_vec) = total_test_num(i_vec)/num_runs(i_vec);
        %ƽ��������
        ave_operation_num(i_vec) = total_operation_num(i_vec)/(num_runs(i_vec)*k);
        
        %���ﵽ������֡��������
        if error_blocks(i_vec) >= max_err*N_u
            break;
        end
        
        %ÿ����display_interval֡���������д������һ��
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
disp(['�������ʱ�䣺' datestr(now)]);
fprintf(filename,'��������ʱ�䣺%s \n',num2str(toc));
fprintf(filename,'�������ʱ�䣺%s \n',datestr(now));
fclose(filename);