% �״���ٲ����
clc;clear;close all;

%% ��������
% ��������
c = 3.0e8; % ����(m/s)
Fc = 35e9; % �״���Ƶ
Br = 10e6; % �����źŴ���
fs = 20*1e6; % ����Ƶ��
PRF = 2e3; % �����ظ�Ƶ��
PRT = 1/PRF; % �����ظ�����
lamda = c/Fc; % �״﹤�����������ڼ��������Ƶ��
N_pulse = 128; % �ز�������
N_sample = round(fs*PRT); % ÿ���������ڵĲ���������
Tr = 3*1e-6; % �����ź�ʱ��
t1 = (0:1/fs:(N_sample-1)/fs); % ʱ������
RangeMax = c*t1(end)/2;% ���ģ������
Range = c*t1/2; % ��������
Vmax = lamda*PRF/2; % ���ɼ���ٶ�
Velocity = -Vmax/2:Vmax/N_pulse:Vmax/2-Vmax/N_pulse; % �ٶ�����
searching_doa = -15:0.01:15; % �Ƕ���������

% ���в���
M = 16; % ��Ԫ����
SourceNum = 1; % �ź�Դ����
d = lamda/2; % ��Ԫ���
d_LinearArray = (0:M-1)'*d; % ��Ԫ���
SNR = 10;
SNR = 10^(SNR/10); % ���ӱ� Pt=10w RCS=20m^2

% ��������
V = 50;
T = 1; % ���������û�в���PRT����Ϊ��T=PRTʱ��nT=300001,��һ�δ���̫��
nT = size((-4e3:V*T:4e3),2); % ����֡��
Xk = [-4e3:V*T:4e3; 64e3*ones(1,nT); V*ones(1,nT); zeros(1,nT)]; % Ŀ��״̬�仯
for i = 1:nT
    r = sqrt(Xk(1,i)^2 + Xk(2,i)^2); % �������
    v = -(Xk(1,i) * Xk(3,i) + Xk(2,i) * Xk(4,i)) / r; % �����ٶ�
    phi = -(atan2d(Xk(2,i), Xk(1,i))-90); % �Ƕ�
    Zk(:,i) = [r;v;phi]; % iʱ��Ŀ�꼫����״̬
end

%% �����������ߵ�����
theta = -90:0.01:90;
theta1 = -3;             %����Aָ��ķ��򣨶ȣ�
theta2 = 3;             %����Bָ��ķ���
theta_min = -3.8;
theta_max = 3.8;
look_a = exp(1j*2*pi*d_LinearArray*sind(theta)/lamda);    %����ʸ��
w_1 = exp(1j*2*pi*d_LinearArray*sind(theta1)/lamda); %����A��ȨȨ����
w_2 = exp(1j*2*pi*d_LinearArray*sind(theta2)/lamda); %����B��ȨȨ����
yA = abs(w_1'*look_a);                                %����A�ķ���ͼ
yB = abs(w_2'*look_a);                                %����B�ķ���ͼ
ABSum = yA+yB;                                    %�Ͳ����ķ���ͼ
ABDiff = yA-yB;                                   %����ķ���ͼ
AB_ybili = ABDiff./ABSum;                              % ��ͱ�
% ����������
figure(1);
plot(theta,(yA/max(yA)),'linewidth',1);   %���Ʋ���A
hold on;
plot(theta,(yB/max(yB)),'linewidth',1);   %���Ʋ���B
xlabel('��λ��/��');
ylabel('��һ������ͼ');
legend('����A','����B');
title('����A��Bʾ��ͼ');
axis tight;
grid on;
% ���ƺͲ��
figure(2);
plot(theta,ABSum,'linewidth',1);   %���ƺͲ���
hold on;
plot(theta,ABDiff,'linewidth',1);   %���Ʋ��
xlabel('��λ��/��');
ylabel('��������');
legend('�Ͳ���','���');
title('�Ͳ��ʾ��ͼ');
axis tight;
grid on;
% ���Ƽ�������
figure(3);
plot(theta,AB_ybili);
xlim([theta_min theta_max]);
xlabel('��λ��/��');
ylabel('��ͱ�');
title('��������');
grid on;

%% ������

%��ʼ������
Detect_Result = zeros(3,nT); % ���ղ������

signal_LFM = zeros(M,N_pulse,N_sample); % �źž���

signal_i = ones(N_pulse,N_sample); % �м��ۼӾ������
y1_out = ones(N_pulse,N_sample);
y2_out = ones(N_pulse,N_sample);

FFT_y1out_all = ones(N_pulse,N_sample,nT); % ����ÿ��nTʱ����MTD��CFAR���
FFT_y2out_all = ones(N_pulse,N_sample,nT);
RDM_mask_A_all = ones(N_pulse,N_sample,nT);  
RDM_mask_B_all = ones(N_pulse,N_sample,nT);

%ƥ���˲�ϵ������
sr = rectpuls(t1-Tr/2,Tr).*exp(1j*pi*(Br/Tr).*(t1-Tr/2).^2);%LFM�����ź�
win = hamming(N_sample)'; %ƥ���˲��Ӵ�
win2 = repmat(hamming(N_pulse),1,N_sample);  %MTD�Ӵ�
h_w = fliplr(conj(sr)).*win;
h_w_freq = fft(h_w);

%����
clutter = sqrt(2)/2*randn(M,N_sample)+sqrt(2)/2*1i*randn(M,N_sample); % ����

for t = 1:nT
    data = Zk(:,t); % ��ȡĿ����ʵλ��
    a_tar_LinearArray = exp(1j*2*pi*d_LinearArray*sind(data(3))/lamda); % �����źŵĵ���ʸ������������

    for i_n = 1:N_pulse
        ta = (i_n-1)*PRT;
        tao = 2*(data(1)-data(2).*(ta+t1))/c;
        signal_i(i_n,:) = SNR.*rectpuls(t1-tao-Tr/2,Tr).*exp(1j*2*pi*Fc*(t1-tao-Tr/2)+1j*pi*(Br/Tr).*(t1-tao-Tr/2).^2);

        signal_LFM(:,i_n,:) = a_tar_LinearArray * signal_i(i_n,:) + clutter;
        st = squeeze(signal_LFM(:,i_n,:));

        y1 = w_1'*st;                                %����A�ز�
        y2 = w_2'*st;                                %����B�ز�

        % ����ѹ��
        y1_out(i_n,:) = ifft(fft(y1,N_sample,2).*h_w_freq,N_sample,2);
        y2_out(i_n,:) = ifft(fft(y2,N_sample,2).*h_w_freq,N_sample,2);

    end

    %% MTD
    FFT_y1out = fftshift(fft(y1_out.*win2),1);
    FFT_y2out = fftshift(fft(y2_out.*win2),1);
    FFT_y1out_all(:,:,t) = FFT_y1out;
    FFT_y2out_all(:,:,t) = FFT_y2out;

    % figure
    % mesh(abs(FFT_y1out))
    % mesh(Range,Velovity,abs(FFT_y1out))
    % figure
    % mesh(abs(FFT_y2out))
    % mesh(Range,Velovity,abs(FFT_y2out))
    %% CA-CFAR

    numGuard = 2; % # of guard cells
    numTrain = numGuard*2; % # of training cells
    P_fa = 1e-5; % desired false alarm rate 
    SNR_OFFSET = -5; % dB

    RDM_dB_y1 = 10*log10(abs(FFT_y1out)/max(max(abs(FFT_y1out))));
    RDM_dB_y2 = 10*log10(abs(FFT_y2out)/max(max(abs(FFT_y2out))));

    % �Բ��� A �Ͳ��� B �ֱ�ִ�� CA-CFAR ���
    [RDM_mask_A, cfar_ranges_A, cfar_dopps_A, K_A] = ca_cfar(RDM_dB_y1, numGuard, numTrain, P_fa, SNR_OFFSET);
    [RDM_mask_B, cfar_ranges_B, cfar_dopps_B, K_B] = ca_cfar(RDM_dB_y2, numGuard, numTrain, P_fa, SNR_OFFSET);
    RDM_mask_A_all(:,:,t) = RDM_mask_A;
    RDM_mask_B_all(:,:,t) = RDM_mask_B;

    %�о�cfarû��ѣ�RDM_mask_A��B���ڼ�����
    cfar_ranges_A = cfar_ranges_A + 1;
    cfar_ranges_B = cfar_ranges_B + 1;
    cfar_dopps_A = cfar_dopps_A + 1;
    cfar_dopps_B = cfar_dopps_B + 1;
    TrgtR = Range(cfar_dopps_A);
    TrftV = Velocity(cfar_ranges_A);

    % ��ȡ��ӦĿ���ڲ��� A �� B �е�ǿ��
    intensity_A = abs(FFT_y1out(cfar_ranges_A, cfar_dopps_A));
    intensity_B = abs(FFT_y2out(cfar_ranges_B, cfar_dopps_B));
        
    % ����ͣ������Ͳ����
    sum_val = intensity_A + intensity_B;
    diff_val = intensity_A - intensity_B;

    % ����Ͳ�� ��/��
    sum_diff_ratio = diff_val / sum_val;
        
    % ���ݺͲ�ȹ��ƽǶȣ�lookup_angle_from_sumdiffratioΪ���ұ����˵ӳ�亯����
    TrgtAngle = lookup_angle_from_sumdiffratio(AB_ybili, sum_diff_ratio, theta, theta_min, theta_max); 
    
    %���²����ٲ�ǽ��
    TrgtInform = [TrgtR;TrftV;TrgtAngle];
    Detect_Result(:,t) = TrgtInform;
end

% ��������
r_all = Detect_Result(1,:);
theta_all = Detect_Result(3,:)+90;
xk_out = [r_all.*cosd(theta_all);r_all.*sind(theta_all)];

RMSE_R_ave = mean(abs(Detect_Result(1,:)-Zk(1,:)));
RMSE_V_ave = mean(abs(Detect_Result(2,:)-Zk(2,:)));
RMSE_phi_ave = mean(abs(Detect_Result(3,:)-Zk(3,:)));

fprintf('ƽ��������%0.2f m \n',RMSE_R_ave);
fprintf('ƽ���������%0.3f m/s \n',RMSE_V_ave);
fprintf('ƽ��������%0.4f �� \n',RMSE_phi_ave);

%% ��ͼ
figure(4); hold on;
plot(d_LinearArray,zeros(1,M),'g^','LineWidth',1.1);
plot(Xk(1,:),Xk(2,:),'b--','LineWidth',1.1);
plot(xk_out(1,:),xk_out(2,:),'rx','LineWidth',1.1);
legend('�״�λ��','��ʵ����','�㼣���ƽ��','Location','southeast');
xlabel('X��m)');ylabel('Y��m)');title('���������');

figure(5); hold on;axis equal;
plot(Xk(1,:),Xk(2,:),'b--','LineWidth',1.1);
plot(xk_out(1,:),xk_out(2,:),'rx','LineWidth',1.1);
legend('��ʵ����','�㼣���ƽ��','Location','southeast');
xlabel('X��m)');ylabel('Y��m)');title('�����Ŵ�ͼ');

figure(6);
mesh(Range,Velocity,abs(FFT_y1out_all(:,:,40)));hold on;
xlabel('���루m)');ylabel('�ٶȣ�m/s)');title('MTD-����A-�����ٶȼ��');
set(gca, 'YDir', 'reverse');

figure(7);
mesh(Range,Velocity,abs(FFT_y2out_all(:,:,40)));hold on;
xlabel('���루m)');ylabel('�ٶȣ�m/s)');title('MTD-����B-�����ٶȼ��');
set(gca, 'YDir', 'reverse');

figure(8);
mesh(Range,Velocity,abs(RDM_mask_A_all(:,:,40)));hold on;
xlabel('���루m)');ylabel('�ٶȣ�m/s)');title('CFAR-����A-�����ٶȼ��');
set(gca, 'YDir', 'reverse');

figure(9);
mesh(Range,Velocity,abs(RDM_mask_B_all(:,:,40)));hold on;
xlabel('���루m)');ylabel('�ٶȣ�m/s)');title('CFAR-����B-�����ٶȼ��');
set(gca, 'YDir', 'reverse');

plotTrajectory(Detect_Result);  % ���ƺ����Ķ�̬��ʾ


function [RDM_mask, cfar_ranges, cfar_dopps, K] = ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET)
    % e.g. numGuard =2, numTrain =2*numGuard, P_fa =1e-5, SNR_OFFSET = -15
    numTrain2D = numTrain*numTrain - numGuard*numGuard;
    RDM_mask = zeros(size(RDM_dB));
    
    for r = numTrain + numGuard + 1 : size(RDM_mask,1) - (numTrain + numGuard)
        for d = numTrain + numGuard + 1 : size(RDM_mask,2) - (numTrain + numGuard)
            
            Pn = ( sum(sum(RDM_dB(r-(numTrain+numGuard):r+(numTrain+numGuard),d-(numTrain+numGuard):d+(numTrain+numGuard)))) - ...
                sum(sum(RDM_dB(r-numGuard:r+numGuard,d-numGuard:d+numGuard))) ) / numTrain2D; % noise level
            a = numTrain2D*(P_fa^(-1/numTrain2D)-1); % scaling factor of T = ��*Pn
            threshold = a*Pn;
            if (RDM_dB(r,d) > threshold) && (RDM_dB(r,d) > SNR_OFFSET)
                RDM_mask(r,d) = 1;
            end
        end
    end
    
    % figure(2)
    % imagesc(RDM_mask);
    % title('CA-CFAR')
    
    [cfar_ranges, cfar_dopps]= find(RDM_mask); % cfar detected range bins
    
    %% remaining part is for target location estimation
    rem_range = zeros(1,length(cfar_ranges));
    rem_dopp = zeros(1,length(cfar_dopps));
    for i = 2:length(cfar_ranges)
       if (abs(cfar_ranges(i) - cfar_ranges(i-1)) <= 5) && (abs(cfar_dopps(i) - cfar_dopps(i-1)) <= 5)
           rem_range(i) = i; % redundant range indices to be removed
           rem_dopp(i) = i; % redundant doppler indices to be removed
       end
    end
    rem_range = nonzeros(rem_range); % filter zeros
    rem_dopp = nonzeros(rem_dopp); % filter zeros
    cfar_ranges(rem_range) = [];
    cfar_dopps(rem_dopp) = [];
    K = length(cfar_dopps); % # of detected targets
end

function theta_closest = lookup_angle_from_sumdiffratio(AB_ybili, sum_diff_ratio, theta, theta_min, theta_max)
    % ��� AB_ybili �� theta �ĳ����Ƿ�һ��
    if length(AB_ybili) ~= length(theta)
        error('AB_ybili �ĳ��ȱ����� theta �ĳ���һ��');
    end

    % ���Ʋ��ҷ�ΧΪ [theta_min, theta_max]
    theta_range_idx = (theta >= theta_min) & (theta <= theta_max);
    theta_limited = theta(theta_range_idx);
    AB_ybili_limited = AB_ybili(theta_range_idx);

    % ���� AB_ybili_limited �� sum_diff_ratio �Ĳ�ֵ
    diff = abs(AB_ybili_limited - sum_diff_ratio);

    % �ҵ���ֵ��С��λ��
    [~, idx] = min(diff);

    % ��ȡ��Ӧ�� theta ֵ
    theta_closest = theta_limited(idx);
end

function plotTrajectory(Detect_Result)
    % plotTrajectory ����Ŀ��ļ������ֱ�����궯̬�켣��������ΪGIF
    % ���룺
    %   Detect_Result: 3xN�ľ��󣬵�һ����Ŀ����룬��������Ŀ����y��ĽǶ�
    
    % ��ȡ����
    distance = Detect_Result(1, :);         % ����
    angle_xy = Detect_Result(3, :);         % ��y��н�
    angle_polar = -1 * angle_xy + 90;       % ת��Ϊ�������µĽǶ�
    
    % ������ϵ����
    filename_polar = 'detect_polar.gif';
    h = figure;
    for i = 1:length(distance)
        polarplot(angle_polar(i) * pi / 180, distance(i), 'bo');
        thetalim([0, 180]);
        % thetalim([75, 105]);
        % rlim([63500, 64500]);
        title('������̬��ʾ');
        % thetaticklabels([]);
        % rticklabels([]);
        hold on;
        drawnow;
        
        % Capture the plot as an image 
        frame = getframe(h); 
        im = frame2im(frame); 
        [imind, cm] = rgb2ind(im, 256); 
        if i == 1 
            imwrite(imind, cm, filename_polar, 'gif', 'Loopcount', inf, 'DelayTime', 0.05); 
        else 
            imwrite(imind, cm, filename_polar, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05); 
        end
    end
    close(h);  % �رռ�����ͼ����

    % ֱ������ϵ����
    filename_xy = 'detect_xy.gif';
    X = distance .* sind(angle_xy);  % x ����
    Y = distance .* cosd(angle_xy);  % y ����
    b = figure;
    for i = 1:length(distance)
        plot(X(i), Y(i), 'bo');
        grid on;
        xlabel('ˮƽ���루m��');
        ylabel('��ֱ���루m��');
        axis([-4500 4500 0 70000]);  % ���������᷶Χ
        title('������̬��ʾ');
        hold on;
        drawnow;
        
        % Capture the plot as an image 
        frame = getframe(b); 
        im = frame2im(frame); 
        [imind, cm] = rgb2ind(im, 256); 
        if i == 1 
            imwrite(imind, cm, filename_xy, 'gif', 'Loopcount', inf, 'DelayTime', 0.05); 
        else 
            imwrite(imind, cm, filename_xy, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05); 
        end
    end
    close(b);  % �ر�ֱ������ͼ����
end

