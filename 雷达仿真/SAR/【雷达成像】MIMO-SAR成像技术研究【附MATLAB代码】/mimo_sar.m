%%%%%%%%%%%%%%飞机沿x轴飞行，阵列布置在机翼上，按照模式一：单阵元，然后多阵元叠加
%%%%%%%%%%运动单点，在坐标系中心，多次快拍结果，为正侧视，目标起始X=0，即飞机阵列为中心点
%****************阵列在x轴上，飞机阵列为9m，合成孔径长度为500m，合成阵列为500+10
%%%%%%由于Lc>1.2*sqrt(lamuda*R0)%22.754，合成阵列为远场，但飞机阵列为远场
%%%%%%当距离分辨率为1m时，距离徙动Rq=lamuda^2*Rmin/32=0.1，远小于距离分辨率，不用作包络移动补偿。距离徙动差delta_Rq=lamuda^2*Rmin/32
%%%%%%%每个阵列先进行基带转换
%%%%%%%%%%%%%%%%固定目标现在可以使用
clc;
close all;
clear all;

%%%%%%%%%设置基本参数%%%%%%%%
v=150;            %载机速度
h=10000;           %载机高度

L_sar=500;     %合成孔径阵列的长度
c=3.0e8;          %波速

%%********飞机阵列参数设置*******CAUTION:是否符合采样定理****************************************************
Ltx=9;        %X方向上阵列长度 
num_array_x=9+1;%X方向阵元数  
dt_x=Ltx/(num_array_x-1);%X方向阵元间距
Array_x1=(-(num_array_x)/2:(num_array_x)/2-1)*dt_x;%X轴上阵元的坐标   1*11  Xq
%%%%%%%%%%%%对长阵元而言
% Array_x=Array_x1(num_array_x)+array_X;   %单个阵元，慢时间tm所对应的坐标为
num_L_sar=(L_sar+Ltx)/dt_x+1;                                         %总阵元数
num_Ltx=Ltx/dt_x+1;
Array_x=(-(num_L_sar)/2:(num_L_sar)/2-1)*dt_x;%X轴上阵元的坐标
%%%%%%%%%设置目标坐标点%%%%%%%%

Xc=0;%%目标x轴中心位置
Yc=100000;%%%目标y轴中心位置
X0=300;                  % Target area in range is within [Xc-X0,Xc+X0] 保证目标运动在[-X0,X0],[-Y0,Y0]之中
Y0=300;                  % Target area in cross-range is within [-Y0,Y0]
Xb=10;%%目标x轴半宽度
Yb=10;%%目标y轴半宽度
Xt0=0;


Rmax=sqrt((Yc)^2+h^2+(L_sar/2+Ltx/2)^2);%%%目标中心到阵元中心最远距离
Rmin=sqrt((Yc)^2+h^2);%%%目标中心到阵元中心理论最近距离

%%%%%%%%%%%%%%%%%%%%%%%%slow time
ds=1.5*2*Rmax/c;                 %防止距离模糊
dt=(Ltx+dt_x)/v;                          %根据飞机速度的选取
%%%%%%%两个时间进行对比，选取大的那个作为PRF，要大于如此情况下的dt，必须在10000000外，不可能
flag=0;                  % flag=0 indicates that ds > dt
if ds < dt,
 flag=1;                 % flag=1 indicates that ds < dt
 ds_temp=ds;             % Store dt
 ds=dt;                 % Choose dtc (dtc > dt) for data acquisition
end;

dku=v*ds;                 %%慢时间距离间隔
fr=1/ds;                         %sample spacing in slow-time domain
n_slow=round((L_sar)/v/ds); %sample number in slow-time domain


tm=linspace(-L_sar/2/v,L_sar/2/v,n_slow+1);%slow time  ds*(-n_slow/2:n_slow/2-1)
array_X=linspace(-L_sar/2,L_sar/2,n_slow+1);%discrete center of the array in slow-time domain


%%%%%%%%%%%%%%%%%%%%%%%%%设定目标


xn=[Xc];
yn=[Yc];
% 
Mx=xn(1);
My=yn(1);

Mvx=0;
Mvy=0;
Max=0;
May=0;
%%%%%%%%%%%CAUTION%%%不能超过设定的目标范围,即X0/Y0
for k=2:length(tm)
Mx(k)=Mx(1)+Mvx*(tm(k)-tm(1))+0.5*Max*(tm(k)-tm(1))^2;
My(k)=My(1)+Mvy*(tm(k)-tm(1))+0.5*May*(tm(k)-tm(1))^2;
end
xn=Mx;
yn=My;


%%%%%%%%%%%定义距离校正量
%%%%%%%%点每次到初始位置的斜距
R_NN0=sqrt((yn-yn(1)).^2+(xn-xn(1)).^2);%%%
%%%%%%%%%%%%飞机此过程的位置
Xt=Xt0+v*tm;
Rq=sqrt(Yc^2+h^2+(Xt(1)-Xc)^2);%%%阵元中心到目标中心首次照射距离
%%%%%%%%%每次目标到阵列中心的斜距Rn
Rn=sqrt(yn.^2+h^2+(Xt-xn).^2);
%%%%%%%%%每次的目标到阵列初始中心的斜距R_XN0
R_XN0=sqrt(yn(1)^2+h^2+(Xt-xn(1)).^2);

%%%%%%%%%%角度XN0N
sita_XN0N=acos((R_XN0.^2+R_NN0.^2-Rn.^2)./R_NN0./R_XN0/2);
sita_XN0N=[0 sita_XN0N(2:n_slow+1)];



%%********发射信号参数设置*************正侧视
fc=11.8e9;%%载频X波段
lamuda=c/fc;%波长
kc=2*pi/lamuda;      %相当于Wc空域
range_resolution=1;%%系统距离分辨力
width_subpulse=2*range_resolution/c;%%编码子脉冲宽度
band_frequence=1/width_subpulse;%%对应频域谱宽
Dopler_frequency=[];
Dopler_frequency=100000;%%100KHz

%%********确定距离域采样数 快时间*************正侧视
num_sample_range_max=180;%距离域采样数 
dieta_f=band_frequence/num_sample_range_max;%%对应频域采样间隔
f_range=(0:num_sample_range_max-1)'*dieta_f;%128*1
time=(0:num_sample_range_max-1)*width_subpulse*c/2/range_resolution;%%转换为距离（单位km）


%********谱坐标*********
dkuc_x=2*pi/(Ltx+L_sar+1);      %Y方向上谱坐标单元
kuc_x=dkuc_x*(-(num_L_sar)/2:(num_L_sar)/2-1);
kuc_x_all=ones(100,1)*kuc_x;
% dkuc_x=2*pi/(L_sar+Ltx);      %X方向上谱坐标单元
% kuc_x=dkuc_x*(-(num_L_sar-1)/2:(num_L_sar-1)/2); %X方向上阵元的谱坐标 1*1001
%%%%%%%%%%基带转换角度XMＮ２
zero_sita_x=atan(-(Xt-xn)./sqrt(h^2+yn.^2));%%%%%%%%abs
Tkus_fai=2*kc*sin(zero_sita_x);

%Return Wave Data
M=num_sample_range_max;                      %Length of fast time   n_fast
N=num_L_sar;                      %Length of slow time
N_slow=num_L_sar;

s=zeros(M,Ltx);                %Initialize memorizer s，两个阵元的间隔
s_out=zeros(M,N);
w=chebwin(length(tm)).';
signal=0;
  signal_recieve=[];%%构造距离向采样信号
for t=1:length(tm)        %%%%%%%按照慢时间量来进行信号叠加
%  for r=1:num_array_x      %%%%按照每个阵元的循环相加length(num_array_x) 
%%%%%%%          Array_x1(r)+Xt慢时间阵元的位置
    dis=[];
    dis=2*sqrt((Array_x1+array_X(t)-xn(t)).^2+(yn(t))^2+h^2)-2*Rmin+range_resolution*num_sample_range_max;%%计算目标至收发阵列的距离,并进行截取 t-2*R/c   1*N 
    
 %%%%%%由于假设全向天线，全程都可以照射
       for ii=1:length(dis)
    tao=(dis(ii))/c;          %%%时延信号   
    signal_delay=exp(-1i*2*pi*f_range*tao);%时延函数   
       temp_signal=[];
        temp_signal=ifft(signal_delay)*exp(-1i*kc*dis(ii));%128*1 加入方位向的信息*w(ii)
        signal_recieve=[signal_recieve temp_signal];
    end
end
figure(1)

row=time-time(180)+Rmin;
col=Array_x+0.5;
contour(col,row,abs(signal_recieve))
% colormap(gray)
ylabel('Range (m)')
xlabel('Cross-range (m)')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                       阵元间距校正                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Approximate_distance=[];
for n2=1:length(Array_x)
    temp1=(Array_x(n2))*(xn(1))/Rmin;
    temp2=0.5*((Array_x(n2))^2)*(Rmin^-1-(xn(1))^2/Rmin^3);
    Approximate_distance=[Approximate_distance 2*(-temp1+temp2)];
end
% Approximate_distance2=Approximate_distance-2*Rmin;%%距离走动与弯曲量之和
s_adjusted=[];
for n3=1:length(Approximate_distance)
    adjusted_function=[];adjusted_function=exp(1i*2*pi*f_range*(Approximate_distance(n3))/c);
    s_temp=[];s_temp=signal_recieve(:,n3);
    temp_adjusted=[];temp_adjusted=fft(s_temp).*adjusted_function;
    s_adjusted=[s_adjusted ifft(temp_adjusted)];
end
signal=s_adjusted;
% range=2*Rmin+range_resolution*N_fast;

figure(4)
row=time-time(180)+Rmin;
col=Array_x+0.5;
contour(col,row,abs(signal))
ylabel('Range (m)')
xlabel('Cross-range (m)')
figure;plot(col,real(signal(91,:)))
xlabel('Cross-range (m)')
ylabel('Real Part')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                      方位向成像                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%3.2方位成像
i2=[]; T_imaging_vector=[];
jishuqi2=0;
%%%%%%%%%%%将数据进行扩展，以对应单独的采样
H00=sqrt(h^2+yn.^2);
H11=repmat(H00,num_array_x,1);
H00=reshape(H11,1,[]);

Xt1=repmat(Xt,num_array_x,1);
Xt=reshape(Xt1,1,[]);

xn1=repmat(xn,num_array_x,1);
xn=reshape(xn1,1,[]);

Tkus_fai1=repmat(Tkus_fai,num_array_x,1);
Tkus_fai=reshape(Tkus_fai1,1,[]);

XN=xn-xn(1);

for i2=1:M
    jishuqi2=jishuqi2+1;
    s_temp=[];s_temp=signal(i2,:);
%     ss=[];ss=s_temp.*exp(-1i*Tkus_fai.*Array_x);  % Baseband conversion for squint
    fs=[];fs=fty(s_temp);                      %b变换到Ku_T域
    %%% NOTE: Tkus_fai (squint Doppler shift) for true ku values
    kx_t=4*(kc^2)-(kuc_x).^2;  %+Tkus_fai
    kx_t=sqrt(kx_t.*(kx_t>0));% kx array
    
    fs0=(kx_t> 0).*exp(1i*kx_t.*H00); % reference signal-Xt +Tkus_fai   +1i*(kuc_x).*(xn))
    F=[];F=fs.*fs0;     % Slow-time matched filtering
    f=[];f=ifty(F);
    T_imaging_vector=[T_imaging_vector;f];

end
 
figure(7)
row=time-time(180)+Rmin;
col=Array_x+0.5;
contour(col,row,abs(T_imaging_vector))
ylabel('Range (m)')
xlabel('Cross-range (m)')
