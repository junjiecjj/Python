clear all;	close all; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   CSA�����㷨 	
%  	For Stanley Park & city,    		use range = 1850, azimuth =7657
% 	first_rg_cell 	 =  1850;    	% Define the range limits
%	first_rg_line  	 =  7657;    	% Define the azimuth limits (19432 max)
%	Nrg_cells    	 =  2048    	% Suggest 2048 cells
%	Nrg_lines_blk  =  1536;   		% Suggest 1536, it should be larger than the
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run specify_parameters.m
% run extract_data.m

%   �������˵��
Fs=32.317e+006;                 	%������,MHz	
Fr=7.2135e+11   ;               	%�����Ƶ��,MHz/us	
start=6.5956e-003;              	%���ݴ���ʼʱ��,ms	
Tr=4.175e-05 ;                  	%����,us	
R0=9.88646462e+05   ;  			    %���б��,m
f0=5.3e+09  ;                   	%�״�Ƶ��,GHz	
lamda=0.05667;                  	%�״ﲨ��,m	
Fa=1.25698e+03 ;                	%�����ظ�Ƶ��, Hz 
Vr=7062;                        	%��Ч�״�����,m/s	
Kr=0.72135e+012;
Ka=1733;                        	%��λ��Ƶ��,Hz/s	
Fc=-6900;                       	%����������Ƶ��,Hz	
c=299790000;                    	%��Ų������ٶ�

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  	Ԥ�ȣ���ȡ����,���ݲ�����վ�ṩ�Ĵ��룬����ȡ����Ҫ�ĳ�������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load CDdata1.mat
data=double(data);					%������ת����double��
[length_a,length_r]=size(data);		%������ݵĴ�С

T_start=6.5956e-003;				%���ݴ���ʼʱ��
tau=T_start:double(1/Fs):T_start+double(length_r/Fs)-double(1/Fs);	%������ʱ��

R_ref=(T_start+length_r/Fs)/2*c;		%�ο�����
f_a=(-Fa/2+Fc):(Fa/length_a):(Fa/2+Fc-Fa/length_a);				%��λƵ��
f_r=0:Fs/length_r:Fs-Fs/length_r;								%����Ƶ��
D = (1 - (f_a*lamda/2/Vr).^2).^0.5; 	%�����㶯����
alpha = 1./D - 1;                  	
R = R_ref./D;                  	%������������и���ȷ��˫���߾����ʽ
 Z=(R0*c*f_a.^2)./(2*Vr^2*f0^3.*D.^3);
Km=Kr./(1-Kr.*Z);					%У������������Ƶ��

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	STEP1:	��λ����Ҷ�任���������źű任�������������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length_r
    data(:,i)=fft(data(:,i));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	STEP2:	���������������ź� �� ���Ա�귽�����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tau1=ones(length_a,1)*tau;
tau2=2.*(R(:)*ones(1,length_r))./c;
D_tau=tau1-tau2;
H1=Km.*alpha;
H1=H1(:)*ones(1,length_r);
Ssc=exp(-j*pi*H1.*D_tau.^2);			% ���Ա�귽��

data=data.*Ssc;					%У������RCM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	STEP3:	��������Ҷ�任���Ӿ����������任����άƵ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length_a
    data(i,:)=fftshift(fft(fftshift(data(i,:))));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	STEP4: 	ͨ��һ����λ��ˣ���ɣ�����ѹ����SRC��һ��RCMC��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

H_r_1=1./(Km.*(1+alpha));
H_r_1=H_r_1(:)*ones(1,length_r);
H_r_2=ones(length_a,1)*f_r.^2;
H_r=exp(-j*pi*H_r_1.*H_r_2);
H_RCMC=exp(j*4*pi*R_ref.*(ones(length_a,1)*f_r).*(alpha(:)*ones(1,length_r))/c);  

data=data.*H_r.*H_RCMC;			%����ѹ����һ��RCMC

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	STEP5: 	��������Ҷ��任����ص������������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length_a
    data(i,:)=fftshift(ifft(fftshift(data(i,:))));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	STEP6: 	���  ��λУ��+��λ��ƥ���˲�
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r_0=tau/2*c;
H3_1=Km.*alpha.*(1+alpha)./(c^2);
H3=H3_1(:)*ones(1,length_r);
phi=4*pi.*H3.*(ones(length_a,1)*(r_0-R_ref).^2)/(c^2);
data=data.*exp(j*phi);				% �����λУ��

H_a= exp(j*4*pi/lamda*(ones(length_a,1)*r_0).*((D(:)-1)*ones(1,length_r)));
data = data .*H_a;					% ��λ��ƥ���˲�

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	STEP7: 	��ɷ�λ����Ҷ��任
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length_r
    data(:,i)=ifft(data(:,i));
end
data = fftshift(data,2);            

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	STEP8: 	��ͼ��۽���ʾ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
colormap(gray(256))
imagesc(log(10*abs(data)));xlabel('Range');ylabel('Azimuth');
