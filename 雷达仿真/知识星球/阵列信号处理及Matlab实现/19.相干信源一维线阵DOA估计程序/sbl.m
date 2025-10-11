clc;
% clear all;
close all;
%%----Remark: the array is place in the range of (fw=-30~30)-----注意：该数组位于（fw=-30~30）范围内
SNR=20;
snr1=10.^(SNR(1)./20);
kp=200;      %快拍数


theta_fy=[35 20  60]; % 0<fy<90 俯仰角
theta_fw=[-30 10 40]; %Incident Direction    入射方向 方位角
theta_gama=[50 35  70 ];%极化辅助角
theta_yita=[60 5   50];%极化相位差
%%%生成网格？
fy=ones(1,181)*90;   %ones函数是生成一个1行181列的全90矩阵
fw=[0:180];   % Note that, the fw should be (-90~90)
gama=[0:0.5:90 ];    %从0到90以0.5为步长进行取值
yita=[0:180];
%角度转化为弧度
rad=pi/180;
rfy=fy*rad;
rfw=fw*rad;
rgama=gama*rad;
ryita=yita*rad;
%%-----polarisation parameter--if there are 2
%%signals----------------------极化参数   如果有两个信号
K1=diag([rgama]);
K2=diag([ryita]);
bc_fy=1; %仰角步长
bc_fw=1;   %方位角步长
NN=90/bc_fy+1; %采样点 bc=1 N=181   NN=91
N=180/bc_fw+1;  %181-1度 91-1度
num_angle=length(fy);    %返回fy中行数或列数中的较大值，即返回181
%生成柱形共形矢量阵列：为一层五个阵元的圆柱阵
sm1=1;      % subarray antenna number   子阵列天线数  long z-dix direction    沿z-dix方向
sm2=5;      % the number along the UCA at the bottom    圆阵列UCA天线数量
sm=sm1*sm2;      %total antenna number    总天线数
M=sm;            %总阵元数
sm3=(sm1)*(sm2);
c=3*10^8;  % light speed   光速
f=1*10^9;   % frequency    频率
wl=c/f;     %波长lamda
dd=wl/2;    % antena space     天线间距
rr=3*wl;    % cylinder radius    圆柱半径


% Trials=3;   %信源数
% epsilon=1;
% thetanum=NN;     %grid    网格
% thetanum1=N; 
% gamma=ones(thetanum,Trials);%91*3 all one
% sigma_w2=1;      %Noise Variance    噪声方差
% sigma_e2=100*sigma_w2;   %100
% sigma_w2=1;      %Noise Variance    噪声方差
% sigma_e2=100*sigma_w2;   %100
% NoiseVector=zeros(M,kp);   % 5*200 all zero   kp是200，M是5
% Ps=sigma_w2*SNR;     %Incident Power    入射功率20
% source_amplitude = sqrt(Ps)*ones(num_angle,kp);                     %Source Amplitude    源振幅  sqrt是计算平方根的函数   ones函数是生成181行200列的全1矩阵
% source_wave = sqrt(0.5)*(randn(kp,num_angle) + j*randn(kp,num_angle)); %Source Wave      源波   randn函数是产生均值为0，方差 σ^2 = 1，标准差σ = 1的正态分布的随机数或矩阵的函数。
% SignalVector=source_amplitude.*source_wave.';       %Source Matrix    源矩阵  Signal Vector  信号矢量

%阵元的横纵坐标
dix=[rr*cos(-pi/6) rr*cos(-pi/12) rr rr*cos(pi/12) rr*cos(pi/6)];%每个阵元的横坐标
diy=[rr*sin(-pi/6) rr*sin(-pi/12) 0  rr*sin(pi/12) rr*sin(pi/6)];%每个阵元的纵坐标

for i=1:sm1
    diz(i)=(i-1)*dd; 
end
for m=1:sm2;
    for n=1:num_angle;
        Ac(m,n)=exp(-j*2*pi/wl*((dix(m))*sin(rfy(n))*cos(rfw(n))+(diy(m))*sin(rfy(n))*sin(rfw(n))));   %空域信息
    end
end
for mm=1:sm1;
    for nn=1:num_angle;
        Az(mm,nn)=exp(-j*2*pi/wl*(diz(mm)*cos(rfy(nn))));    %空域信息diz=0
    end
end
%%----construct polarization matrix-------------构建极化矩阵
%rfw(n)=pi/2;
for n=1:num_angle;
    Ek=[-sin(rfy(n)) cos(rfw(n))*cos(rfy(n));cos(rfy(n)) cos(rfw(n))*sin(rfy(n));0 -sin(rfw(n))];%电场 矢量   极化域信息
    Hk=[cos(rfw(n))*cos(rfy(n)) sin(rfy(n));cos(rfw(n))*sin(rfy(n)) -cos(rfy(n));-sin(rfw(n)) 0];          %极化域信息
    Dk=[Ek;Hk];
    Wk=[cos(rgama(n));sin(rgama(n))*exp(j*(ryita(n)))];
    %    Wk=[sin(rgama(n))*exp(j*(ryita(n)));cos(rgama(n))];
    P1(:,n)=Dk*Wk;%p1为空域极化导向矢量
end
%%----sensor pattern-------------传感器模式
%%-----(azimuth and elvation) global to local coordinate transform(refer to
%%Wang Buhong_Dianzi Xue Bao)---------方位角和俯仰角  全局到局部坐标变换  
D1=150*rad;    % first rotation angle of 1st array   第一个矩阵的第一个旋转角
D2=165*rad;    % first rotation angle of 2nd array   第二个矩阵的第一个旋转角
D3=180*rad;    % first rotation angle of 3rd array   第三个矩阵的第一个旋转角
D4=195*rad;    % first rotation angle of 4th array   第四个矩阵的第一个旋转角
D5=210*rad;    % first rotation angle of 5th array   第五个矩阵的第一个旋转角
E=-pi/2;         % second rotation angle    第二个旋转角
F=0;            % third rotation angle      第三个旋转角
%用欧拉旋转变换求局部坐标系（x',y',z')
%--local coordinate(x',y',z')---(two signals corresponding to 5 subarray)-----局部坐标（x',y',z'）--两个信号对应5个子阵列
F1=zeros(3,num_angle);F2=zeros(3,num_angle);F3=zeros(3,num_angle);F4=zeros(3,num_angle);F5=zeros(3,num_angle);
for rr=1:num_angle                                                                                             %全局坐标系(x,y,z)                   
    F1(:,rr)=[cos(E)*cos(D1) cos(E)*sin(D1) -sin(E);-sin(D1) cos(D1) 0;sin(E)*cos(D1) sin(E)*sin(D1) cos(E)]*[sin(rfy(rr))*cos(rfw(rr));sin(rfy(rr))*sin(rfw(rr));cos(rfy(rr))];
    F2(:,rr)=[cos(E)*cos(D2) cos(E)*sin(D2) -sin(E);-sin(D2) cos(D2) 0;sin(E)*cos(D2) sin(E)*sin(D2) cos(E)]*[sin(rfy(rr))*cos(rfw(rr));sin(rfy(rr))*sin(rfw(rr));cos(rfy(rr))];
    F3(:,rr)=[cos(E)*cos(D3) cos(E)*sin(D3) -sin(E);-sin(D3) cos(D3) 0;sin(E)*cos(D3) sin(E)*sin(D3) cos(E)]*[sin(rfy(rr))*cos(rfw(rr));sin(rfy(rr))*sin(rfw(rr));cos(rfy(rr))];
    F4(:,rr)=[cos(E)*cos(D4) cos(E)*sin(D4) -sin(E);-sin(D4) cos(D4) 0;sin(E)*cos(D4) sin(E)*sin(D4) cos(E)]*[sin(rfy(rr))*cos(rfw(rr));sin(rfy(rr))*sin(rfw(rr));cos(rfy(rr))];
    F5(:,rr)=[cos(E)*cos(D5) cos(E)*sin(D5) -sin(E);-sin(D5) cos(D5) 0;sin(E)*cos(D5) sin(E)*sin(D5) cos(E)]*[sin(rfy(rr))*cos(rfw(rr));sin(rfy(rr))*sin(rfw(rr));cos(rfy(rr))];
end
%%----local theata and fy---------
%将局部直角坐标系(x',y',z')变换为局部极坐标
thta1=zeros(1,num_angle);thta2=zeros(1,num_angle);thta3=zeros(1,num_angle);thta4=zeros(1,num_angle);thta5=zeros(1,num_angle);
fai1=zeros(1,num_angle);fai2=zeros(1,num_angle);fai3=zeros(1,num_angle);fai4=zeros(1,num_angle);fai5=zeros(1,num_angle );
for tt=1:num_angle
    thta1(1,tt)=acos(F1(3,tt));       fai1(1,tt)=atan(F1(2,tt)/F1(1,tt));  %signal in 1st subarray   信号在第一个子阵列   acos函数是反余弦，atan函数是反正弦
    thta2(1,tt)=acos(F2(3,tt));       fai2(1,tt)=atan(F2(2,tt)/F2(1,tt));
    thta3(1,tt)=acos(F3(3,tt));       fai3(1,tt)=atan(F3(2,tt)/F3(1,tt));
    thta4(1,tt)=acos(F4(3,tt));       fai4(1,tt)=atan(F4(2,tt)/F4(1,tt));
    thta5(1,tt)=acos(F5(3,tt));       fai5(1,tt)=atan(F5(2,tt)/F5(1,tt));
end
%-----polarisation componet of local antenna pattern------局部天线图的极化模式

gthta1=sin(thta1-fai1);gfai1=cos(thta1-fai1);  %-----signals to 1st subarray---------信号到第一个子阵列
gthta2=sin(thta2-fai2);gfai2=cos(thta2-fai2);  %-----signals to 2nd subarray---------信号到第二个子阵列
gthta3=sin(thta3-fai3);gfai3=cos(thta3-fai3);  %-----signals to 3rd subarray---------信号到第三个子阵列
gthta4=sin(thta4-fai4);gfai4=cos(thta4-fai4);  %-----signals to 4th subarray---------信号到第四个子阵列
gthta5=sin(thta5-fai5);gfai5=cos(thta5-fai5);  %-----signals to 5th subarray---------信号到第五个子阵列

%-----anteena directin pattern matrix------------------天线方向图矩阵
GGthta=[gthta1;gthta2;gthta3;gthta4;gthta5];
GGfai=[gfai1;gfai2;gfai3;gfai4;gfai5];
Gthta=GGthta.';   %.'的作用是把上面的GGthta行矩阵转化为列矩阵，即取转置
Gfai=GGfai.';  
GG=Gthta.'*K1+Gfai.'*K2;  %  方向图矩阵sensor pattern of two signals  两个信号的传感器模式
%--------------------------------------------------------------------------

AA=kr(Ac,Az);     % steering matrix without sensor pattern 传感器模式的转向矩阵
A0=kr(GG.*Ac,Az);   %kr函数在这里是指调用kr子函数   .*指的是两个矩阵对应每个元素相乘
A00=kr(A0,P1);    %空域极化角度域导向矢量
%%------------------------------------------------
numss=0;
testnum=1:1;
cc=length(testnum);   %返回testnum中行数或列数中的较大值，即返回
anglefw=zeros(cc,num_angle);
anglefy=zeros(cc,num_angle);
for  ii=1:cc;
    
    %------signal source and noise produce-------------信号源与噪声产生
    
    for p=1:num_angle
        for q=1:kp
            p1=rand(1,1);   %rand函数是产生1行1列的一个1以内的随机数
            p2=rand(1,1);
            sr(p,q)=sqrt(-2*snr1*snr1*log(p1))*cos(2*pi*p2);    %real part     sqrt函数是计算平方根的
            si(p,q)=sqrt(-2*snr1*snr1*log(p1))*sin(2*pi*p2);    %image part
            s(p,q)=sr(p,q)+j*si(p,q);    
        end
    end
  
    n1=wgn(sm*6,kp,0,'complex');
  x0=A00*s+n1; % no mutual coupling    无互相耦合
  
d=dix;    %Spacing   阵元间距
T=kp;              %Snapshots   快拍数
thetatest=[-90:90]; 
%Sampling Grid   采样网格
thetanum=length(thetatest);%Sampling Number   抽样数量    thetanum=181
K=size(theta_fw,2);                   %Source Number    源数   size函数是返回theta_fw的第2维的尺寸    K=3
M=sm;                %Array Number    阵元数5
iterNum=10000;      %Iteration Number    迭代数  
epsilon_min=0.0001;     %Threshold     阈值
gamma=ones(thetanum,1);    %gamma是181行1列的全1矩阵181*1
Nsample=length(SNR);     %Nsample好像等于20
estsbl_theta=zeros(Nsample,K);     %estsbl_theta是20行10列的全0矩阵
estmusic_theta=zeros(Nsample,K);   %estmusic_theta是20行10列的全0矩阵
theta_error_sbl=zeros(Nsample,1);  %theta_error_sbl是20行1列的全0矩阵
theta_error_msc=zeros(Nsample,1);  %theta_error_msc是20行1列的全0矩阵
Ntrial=1;
P=ones(NN,N);    %P是91行181列的全1矩阵
for nNsample=1:Nsample    %第一层循环循环20次
    S=SNR(nNsample);
    snr=1;     %Incident Power\\  入射功率
    for n=1:Ntrial     %第二层循环循环1次
        epsilon=1;     %超参数
        sigma_2=1;      %Noise Variance    噪声方差
        Pn=sigma_2;    %Noise Power        噪声功率
        St=zeros(thetanum,T);    %St是181行200列的全0矩阵
        DD1=A00'*ones(6*sm,sm);     %sm=5,
        D=exp(1j*pi*[0:M-1]'*sin(thetatest*pi/180)); %m=5  过完备字典
        source_amplitude = sqrt(sigma_2.*snr)*ones(1,T);                     %Source Amplitude    源振幅   sqrt函数是计算平方根的
        source_wave = sqrt(0.5)*(randn(T,K) + j*randn(T,K)); %Source Wave    源波   randn函数是产生均值为0，方差 σ^2 = 1，标准差σ = 1的正态分布的随机数或矩阵的函数
        SignalVector=source_amplitude.*source_wave.';       %Source Matrix   源矩阵
        NoiseVector = sqrt(0.5*sigma_2)*(randn(M,T)+j*randn(M,T));  %Noise Matrix    噪声矩阵
        %% SBL
        for k=1:K     % k=3第三层循环循环3次
            for i=1:thetanum     %第四层循环循环181次
                if thetatest(i)==theta_fw(k)
                    St(i,:)=SignalVector(k,:);      %Sparse Matrix    稀疏矩阵
                end
            end
        end
        Xt = D*St + NoiseVector;               %Recieved Signal    接收信号
        %% Iteration    迭代
        i=1;
        while (i<iterNum) && (epsilon>epsilon_min)     %条件是迭代数1000，
            pregamma=gamma;                 %Old gamma    gamma是181行1列的全1矩阵
            Gamma=diag(pregamma);           %构造对角矩阵181*181
            sigma_x(:,:)=sigma_2.*eye(M,M)+D*Gamma*D';    % M=5eye函数是单位矩阵函数，主要创建对角元素为1，其他元素为0的矩阵，此处为构造M行M列的矩阵
            U_W(:,:)=Gamma*D'*inv(sigma_x)*Xt;            %Update mean of W   181*200 更新W均值    inv函数是矩阵求逆命令，但应该尽量少使用该命令
            sigma_W=Gamma-Gamma*D'*inv(sigma_x)*D*Gamma;   %Update Covariance of W     更新W的协方差
            for k=1:thetanum              %循环循环181次
                u(k)=norm(U_W(k,:),2);    %求其2范数
                 gamma(k)=u(k)^2/K/(1-sigma_W(k,k)/pregamma(k));   %Update  gamma, i--itteration, k--sampling grid 采样网格   
%                 gamma(k)=u(k)^2/T+sigma_W(k,k);
%                 J(k)=sigma_W(k,k)/gamma(k);%%%#
                J(k)=sigma_W(k,k)/pregamma(k);%%%#这里是gamma(k)还是pregamma(k)存疑！！！
            end
            L=sum(J);
            %             sigma_2=sqrt((Xt-D*U_W)*(Xt-D*U_W)')/M/(M-thetanum+L);
            %             epsilon=norm(gamma-pregamma,inf)/norm(pregamma,inf);
            sigma_2=(Xt-D*U_W)*(Xt-D*U_W)'/K/(M-thetanum+L);     %Update Variance of Noise    更新噪声方差
            epsilon=norm(gamma-pregamma,inf)/norm(pregamma,inf);    %Update Threshold         更新阈值    norm是求范数的函数
            i=i+1;
            estsbl_theta(n,:)=peak_seek(gamma,thetatest,K);      %调用peak_seek子函数
        end
%         for jj=1:K
%             a_sbl(n,jj)=(estsbl_theta(n,jj)-b(jj))^2;
%         end
    end

end

end  

 
  figure(1)    
plot(thetatest,abs(gamma),'linewidth',2)
title('SBL-based DOA Estimation')
xlabel('fw')
ylabel('Spatial spectrum') 

 