clc;
close all;
clear all;

global c;
global j;
c = 3e8;                                        %光速m/s 
j = sqrt(-1);                                   %虚数
%% 雷达参数
radarpara.PRT = 1000e-6;                        %周期s                  
radarpara.Tao = 40e-6;                          %脉冲持续时间 脉宽s
radarpara.B  = 5e6;                             %信号带宽Hz
radarpara.Kr = radarpara.B/radarpara.Tao;       %线性调频率
radarpara.Fs = 2*radarpara.B;                   %距离采样频率24MHz，2为过采样率
radarpara.Ts = 1/radarpara.Fs;                  %距离采样时间间隔s
radarpara.f0 = 3e9;                             %雷达工作频率Hz
radarpara.lamda = c/radarpara.f0;               %雷达工作波长m
radarpara.dx = 0.05;                            %横向单元间距
radarpara.dy = 0.05;                            %纵向单元间距
radarpara.Nx = 40;                              %横向单元数
radarpara.Ny = 40;                              %纵向单元数
radarpara.Nx_main = 1:radarpara.Nx;
radarpara.Ny_main = 1:radarpara.Ny;
radarpara.Nx_slc = 11:18;
radarpara.Ny_slc = 11:18;
radarpara.azi = 0;
radarpara.ele = 0;
radarpara.num_slc = 1;
radarpara.azi_slc = zeros(radarpara.num_slc,1);
radarpara.ele_slc = zeros(radarpara.num_slc,1);
radarpara.azi_slc = [4 ];
radarpara.ele_slc = [4 ];

%
ctrl.Bomen_st = max([60e3,radarpara.Tao*1e6*150]);             %波门起始距离
ctrl.Bomen_end = min([150e3,radarpara.PRT*1e6*150]);           %波门结束距离
ctrl.N = ceil(2 * (ctrl.Bomen_end-ctrl.Bomen_st) / c / radarpara.Ts);  %A/D后总点数
%% 回波
target.R0 = 100e3;                          %目标中心斜距m
target.Azi = radarpara.azi;                                    
target.Ele = radarpara.ele;

false_target.R0 = 110e3;                    %假目标中心斜距m
false_target.Azi = target.Azi;
false_target.Ele = target.Ele;

noise = randn(1,ctrl.N).*exp(j*randn(1,ctrl.N));
echo = noise;

Nst = floor(2*(target.R0-ctrl.Bomen_st) /c / radarpara.Ts) +1;
Ned = floor((2*(target.R0-ctrl.Bomen_st) /c + radarpara.Tao)/ radarpara.Ts) +1;
Ned = min([Ned,ctrl.N]);
t = (1:ctrl.N)*radarpara.Ts+ctrl.Bomen_st/150*1e-6;
r = t*1e6*150;
ts = (Nst:Ned)*radarpara.Ts;
phrase = 1/2*radarpara.Kr*ts.^2;

B = 0 ;
for row = 1:length(radarpara.Nx_main)
    for col = 1:length(radarpara.Nx_main)
         fai = (radarpara.Nx_main(row)-1)*radarpara.dx*sind(target.Azi) + (radarpara.Nx_main(col)-1)*radarpara.dy*sind(target.Ele);
         B = B + exp(-j*2*pi*fai/radarpara.lamda);
    end
end
target1 = B;
echo(Nst:Ned) = echo(Nst:Ned) + target1*exp(j*2*pi*phrase);

%副瓣干扰
Jam.azi1 = radarpara.azi_slc(1);
Jam.ele1 = radarpara.ele_slc(1);
Jam.f1 = radarpara.f0;
B = 0 ;
for row = 1:length(radarpara.Nx_main)
    for col = 1:length(radarpara.Nx_main)
         fai = (radarpara.Nx_main(row)-1)*radarpara.dx*sind(Jam.azi1) + (radarpara.Nx_main(col)-1)*radarpara.dy*sind(Jam.ele1);
         B = B + exp(-j*2*pi*fai/radarpara.lamda);
    end
end
target2 = B;

Nst2 = floor(2*(false_target.R0-ctrl.Bomen_st) /c / radarpara.Ts) +1;
Ned2 = floor((2*(false_target.R0-ctrl.Bomen_st) /c + radarpara.Tao)/ radarpara.Ts) +1;
Ned2 = min([Ned2,ctrl.N]);
ts2 = (Nst2:Ned2)*radarpara.Ts;
phrase2 = 1/2*radarpara.Kr*ts2.^2;
echo(Nst2:Ned2) = echo(Nst2:Ned2) + target2*exp(j*2*pi*phrase2);


figure;
subplot(3,1,1)
plot(r/1e3,db(abs(echo)),'.-');
grid;
xlabel('R/km');ylabel('幅度/dB');
title('主通道信号');

%对消通道
B = 0;
for row = 1:length(radarpara.Nx_slc)
    for col = 1:length(radarpara.Ny_slc)
         fai = (radarpara.Nx_slc(row)-1)*radarpara.dx*sind(Jam.azi1) + (radarpara.Ny_slc(col)-1)*radarpara.dy*sind(Jam.ele1);
         B = B + exp(-j*2*pi*fai/radarpara.lamda);
    end
end
target3 = B;
noise = randn(1,ctrl.N).*exp(j*randn(1,ctrl.N));
echo_slc(1,:) = noise;
echo_slc(1,Nst2:Ned2) = echo_slc(1,Nst2:Ned2) + target3*exp(j*2*pi*phrase2);

subplot(3,1,2)
plot(r/1e3,db(abs(echo_slc(1,:))),'.-');
grid;
xlabel('R/km');ylabel('幅度/dB');
title('对消通道信号');

noise = randn(1,ctrl.N).*exp(j*randn(1,ctrl.N));
echo_slc(2,:) = noise;
echo_slc(2,Nst2:Ned2) = echo_slc(2,Nst2:Ned2) + target3*exp(j*2*pi*phrase2);

subplot(3,1,3)
plot(r/1e3,db(abs(echo_slc(2,:))),'.-');
grid;
xlabel('R/km');ylabel('幅度/dB');
title('对消通道信号2');

figure;
subplot(2,1,1)
plot(Nst:Ned,angle(echo(Nst:Ned)),'.-');
grid;
subplot(2,1,2)
plot(Nst:Ned,angle(echo_slc(Nst:Ned)),'.-');
grid;


angle_var = angle(echo(Nst:Ned))- angle(echo_slc(Nst:Ned));
index = find(angle_var < 2*pi);
angle_var(index) = angle_var(index) + 2*pi;
index = find(angle_var > 2*pi);
angle_var(index) = angle_var(index) - 2*pi;
figure;
plot(1:Ned-Nst+1,angle_var,'.');
grid;

angle_var = angle(echo(Nst2:Ned))- angle(echo_slc(Nst2:Ned));
index = find(angle_var < 2*pi);
angle_var(index) = angle_var(index) + 2*pi;
index = find(angle_var > 2*pi);
angle_var(index) = angle_var(index) - 2*pi;
figure;
plot(1:length(angle_var),angle_var,'.');
grid;


result_slc = slc(radarpara,ctrl,echo,echo_slc);%对消


function result_slc = slc(radarpara,ctrl,echo,echo_slc)

    num_slc = size(echo_slc,1);%对消波束个数
    
    %选干扰样本
    R_starts = 1:128:length(echo);
    Result = zeros(num_slc,length(R_starts)-1,1);
    for j = 1:num_slc
        for i = 1:length(R_starts)-1
            Rx = R_starts(i):R_starts(i+1)-1;
            Result(j,i) = abs( echo(1,Rx)*echo_slc(j,Rx)'  )./(norm(echo(1,Rx))*norm(echo_slc(j,Rx)));
        end
    end
    Result1 = max(Result,[],1);
    index_slc = find(Result1 > 0.5);               
    
    r = ctrl.Bomen_st + R_starts(1:end-1)*radarpara.Ts*1e6*150;
    r = r/1e3;
    figure;
    plot(r,Result,'.-');
    grid;
    xlabel('R/km');
    ylim([-1,2]);
    title('副瓣对消选样本');
    
    %计算权值
    W = 0;
    for i = 1:length(index_slc)
        Rx = R_starts(index_slc(i)):R_starts(index_slc(i)+1)-1;
        sample_echo = echo(1,Rx);
        sample_echo_slc = echo_slc(1,Rx);
        W  =  W + sum(sample_echo.*sample_echo_slc)/length(sample_echo) * inv(sum(sample_echo_slc.*sample_echo_slc)/length(sample_echo));
    end
    W = W/length(index_slc);

%     index =  13280;
%     W  =  echo(1,index)*echo_slc(1,index) * inv(echo_slc(1,index)*echo_slc(1,index));

    
    %副瓣对消
%     Ryy = echo_slc*echo_slc'/length(echo_slc);
%     Rxy = echo*echo_slc'/length(echo);
%     result_slc = echo - abs(Rxy)^2*inv(Rxy)/Ryy*echo_slc;
    result_slc = echo - W*echo_slc;

    t = (1:ctrl.N)*radarpara.Ts+ctrl.Bomen_st/150*1e-6;
    r = t*1e6*150;
    figure;
    plot(r/1e3,db(abs(result_slc)),'.-');
    grid;
    ylim([-20,80]);
end

