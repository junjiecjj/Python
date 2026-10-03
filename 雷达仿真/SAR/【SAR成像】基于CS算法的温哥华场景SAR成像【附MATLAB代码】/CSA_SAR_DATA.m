clear all;	close all; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   CSA成像算法 	
%  	For Stanley Park & city,    		use range = 1850, azimuth =7657
% 	first_rg_cell 	 =  1850;    	% Define the range limits
%	first_rg_line  	 =  7657;    	% Define the azimuth limits (19432 max)
%	Nrg_cells    	 =  2048    	% Suggest 2048 cells
%	Nrg_lines_blk  =  1536;   		% Suggest 1536, it should be larger than the
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run specify_parameters.m
% run extract_data.m

%   成像参数说明
Fs=32.317e+006;                 	%采样率,MHz	
Fr=7.2135e+11   ;               	%距离调频率,MHz/us	
start=6.5956e-003;              	%数据窗开始时间,ms	
Tr=4.175e-05 ;                  	%脉宽,us	
R0=9.88646462e+05   ;  			    %最短斜距,m
f0=5.3e+09  ;                   	%雷达频率,GHz	
lamda=0.05667;                  	%雷达波长,m	
Fa=1.25698e+03 ;                	%脉冲重复频率, Hz 
Vr=7062;                        	%有效雷达速率,m/s	
Kr=0.72135e+012;
Ka=1733;                        	%方位调频率,Hz/s	
Fc=-6900;                       	%多普勒中心频率,Hz	
c=299790000;                    	%电磁波传播速度

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  	预先：读取数据,数据采用网站提供的代码，并提取所需要的成像区域
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load CDdata1.mat
data=double(data);					%将数据转换成double型
[length_a,length_r]=size(data);		%获得数据的大小

T_start=6.5956e-003;				%数据窗开始时间
tau=T_start:double(1/Fs):T_start+double(length_r/Fs)-double(1/Fs);	%距离向时间

R_ref=(T_start+length_r/Fs)/2*c;		%参考距离
f_a=(-Fa/2+Fc):(Fa/length_a):(Fa/2+Fc-Fa/length_a);				%方位频率
f_r=0:Fs/length_r:Fs-Fs/length_r;								%距离频率
D = (1 - (f_a*lamda/2/Vr).^2).^0.5; 	%距离徙动因子
alpha = 1./D - 1;                  	
R = R_ref./D;                  	%距离多普勒域中更精确的双曲线距离等式
 Z=(R0*c*f_a.^2)./(2*Vr^2*f0^3.*D.^3);
Km=Kr./(1-Kr.*Z);					%校正后距离脉冲调频率

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	STEP1:	方位向傅里叶变换，将基带信号变换到距离多普勒域
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length_r
    data(:,i)=fft(data(:,i));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	STEP2:	将距离多普勒域的信号 与 线性变标方程相乘
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tau1=ones(length_a,1)*tau;
tau2=2.*(R(:)*ones(1,length_r))./c;
D_tau=tau1-tau2;
H1=Km.*alpha;
H1=H1(:)*ones(1,length_r);
Ssc=exp(-j*pi*H1.*D_tau.^2);			% 线性变标方程

data=data.*Ssc;					%校正补余RCM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	STEP3:	距离向傅里叶变换，从距离多普勒域变换到二维频域
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length_a
    data(i,:)=fftshift(fft(fftshift(data(i,:))));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	STEP4: 	通过一个相位相乘，完成（距离压缩、SRC、一致RCMC）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

H_r_1=1./(Km.*(1+alpha));
H_r_1=H_r_1(:)*ones(1,length_r);
H_r_2=ones(length_a,1)*f_r.^2;
H_r=exp(-j*pi*H_r_1.*H_r_2);
H_RCMC=exp(j*4*pi*R_ref.*(ones(length_a,1)*f_r).*(alpha(:)*ones(1,length_r))/c);  

data=data.*H_r.*H_RCMC;			%距离压缩和一致RCMC

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	STEP5: 	距离向傅里叶逆变换，变回到距离多普勒域
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length_a
    data(i,:)=fftshift(ifft(fftshift(data(i,:))));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	STEP6: 	完成  相位校正+方位向匹配滤波
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r_0=tau/2*c;
H3_1=Km.*alpha.*(1+alpha)./(c^2);
H3=H3_1(:)*ones(1,length_r);
phi=4*pi.*H3.*(ones(length_a,1)*(r_0-R_ref).^2)/(c^2);
data=data.*exp(j*phi);				% 完成相位校正

H_a= exp(j*4*pi/lamda*(ones(length_a,1)*r_0).*((D(:)-1)*ones(1,length_r)));
data = data .*H_a;					% 方位向匹配滤波

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	STEP7: 	完成方位向傅里叶逆变换
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length_r
    data(:,i)=ifft(data(:,i));
end
data = fftshift(data,2);            

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	STEP8: 	将图像聚焦显示
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
colormap(gray(256))
imagesc(log(10*abs(data)));xlabel('Range');ylabel('Azimuth');
