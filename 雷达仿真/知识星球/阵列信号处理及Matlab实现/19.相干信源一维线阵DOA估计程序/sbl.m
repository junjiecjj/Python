clc;
% clear all;
close all;
%%----Remark: the array is place in the range of (fw=-30~30)-----ע�⣺������λ�ڣ�fw=-30~30����Χ��
SNR=20;
snr1=10.^(SNR(1)./20);
kp=200;      %������


theta_fy=[35 20  60]; % 0<fy<90 ������
theta_fw=[-30 10 40]; %Incident Direction    ���䷽�� ��λ��
theta_gama=[50 35  70 ];%����������
theta_yita=[60 5   50];%������λ��
%%%��������
fy=ones(1,181)*90;   %ones����������һ��1��181�е�ȫ90����
fw=[0:180];   % Note that, the fw should be (-90~90)
gama=[0:0.5:90 ];    %��0��90��0.5Ϊ��������ȡֵ
yita=[0:180];
%�Ƕ�ת��Ϊ����
rad=pi/180;
rfy=fy*rad;
rfw=fw*rad;
rgama=gama*rad;
ryita=yita*rad;
%%-----polarisation parameter--if there are 2
%%signals----------------------��������   ����������ź�
K1=diag([rgama]);
K2=diag([ryita]);
bc_fy=1; %���ǲ���
bc_fw=1;   %��λ�ǲ���
NN=90/bc_fy+1; %������ bc=1 N=181   NN=91
N=180/bc_fw+1;  %181-1�� 91-1��
num_angle=length(fy);    %����fy�������������еĽϴ�ֵ��������181
%�������ι���ʸ�����У�Ϊһ�������Ԫ��Բ����
sm1=1;      % subarray antenna number   ������������  long z-dix direction    ��z-dix����
sm2=5;      % the number along the UCA at the bottom    Բ����UCA��������
sm=sm1*sm2;      %total antenna number    ��������
M=sm;            %����Ԫ��
sm3=(sm1)*(sm2);
c=3*10^8;  % light speed   ����
f=1*10^9;   % frequency    Ƶ��
wl=c/f;     %����lamda
dd=wl/2;    % antena space     ���߼��
rr=3*wl;    % cylinder radius    Բ���뾶


% Trials=3;   %��Դ��
% epsilon=1;
% thetanum=NN;     %grid    ����
% thetanum1=N; 
% gamma=ones(thetanum,Trials);%91*3 all one
% sigma_w2=1;      %Noise Variance    ��������
% sigma_e2=100*sigma_w2;   %100
% sigma_w2=1;      %Noise Variance    ��������
% sigma_e2=100*sigma_w2;   %100
% NoiseVector=zeros(M,kp);   % 5*200 all zero   kp��200��M��5
% Ps=sigma_w2*SNR;     %Incident Power    ���书��20
% source_amplitude = sqrt(Ps)*ones(num_angle,kp);                     %Source Amplitude    Դ���  sqrt�Ǽ���ƽ�����ĺ���   ones����������181��200�е�ȫ1����
% source_wave = sqrt(0.5)*(randn(kp,num_angle) + j*randn(kp,num_angle)); %Source Wave      Դ��   randn�����ǲ�����ֵΪ0������ ��^2 = 1����׼��� = 1����̬�ֲ�������������ĺ�����
% SignalVector=source_amplitude.*source_wave.';       %Source Matrix    Դ����  Signal Vector  �ź�ʸ��

%��Ԫ�ĺ�������
dix=[rr*cos(-pi/6) rr*cos(-pi/12) rr rr*cos(pi/12) rr*cos(pi/6)];%ÿ����Ԫ�ĺ�����
diy=[rr*sin(-pi/6) rr*sin(-pi/12) 0  rr*sin(pi/12) rr*sin(pi/6)];%ÿ����Ԫ��������

for i=1:sm1
    diz(i)=(i-1)*dd; 
end
for m=1:sm2;
    for n=1:num_angle;
        Ac(m,n)=exp(-j*2*pi/wl*((dix(m))*sin(rfy(n))*cos(rfw(n))+(diy(m))*sin(rfy(n))*sin(rfw(n))));   %������Ϣ
    end
end
for mm=1:sm1;
    for nn=1:num_angle;
        Az(mm,nn)=exp(-j*2*pi/wl*(diz(mm)*cos(rfy(nn))));    %������Ϣdiz=0
    end
end
%%----construct polarization matrix-------------������������
%rfw(n)=pi/2;
for n=1:num_angle;
    Ek=[-sin(rfy(n)) cos(rfw(n))*cos(rfy(n));cos(rfy(n)) cos(rfw(n))*sin(rfy(n));0 -sin(rfw(n))];%�糡 ʸ��   ��������Ϣ
    Hk=[cos(rfw(n))*cos(rfy(n)) sin(rfy(n));cos(rfw(n))*sin(rfy(n)) -cos(rfy(n));-sin(rfw(n)) 0];          %��������Ϣ
    Dk=[Ek;Hk];
    Wk=[cos(rgama(n));sin(rgama(n))*exp(j*(ryita(n)))];
    %    Wk=[sin(rgama(n))*exp(j*(ryita(n)));cos(rgama(n))];
    P1(:,n)=Dk*Wk;%p1Ϊ���򼫻�����ʸ��
end
%%----sensor pattern-------------������ģʽ
%%-----(azimuth and elvation) global to local coordinate transform(refer to
%%Wang Buhong_Dianzi Xue Bao)---------��λ�Ǻ͸�����  ȫ�ֵ��ֲ�����任  
D1=150*rad;    % first rotation angle of 1st array   ��һ������ĵ�һ����ת��
D2=165*rad;    % first rotation angle of 2nd array   �ڶ�������ĵ�һ����ת��
D3=180*rad;    % first rotation angle of 3rd array   ����������ĵ�һ����ת��
D4=195*rad;    % first rotation angle of 4th array   ���ĸ�����ĵ�һ����ת��
D5=210*rad;    % first rotation angle of 5th array   ���������ĵ�һ����ת��
E=-pi/2;         % second rotation angle    �ڶ�����ת��
F=0;            % third rotation angle      ��������ת��
%��ŷ����ת�任��ֲ�����ϵ��x',y',z')
%--local coordinate(x',y',z')---(two signals corresponding to 5 subarray)-----�ֲ����꣨x',y',z'��--�����źŶ�Ӧ5��������
F1=zeros(3,num_angle);F2=zeros(3,num_angle);F3=zeros(3,num_angle);F4=zeros(3,num_angle);F5=zeros(3,num_angle);
for rr=1:num_angle                                                                                             %ȫ������ϵ(x,y,z)                   
    F1(:,rr)=[cos(E)*cos(D1) cos(E)*sin(D1) -sin(E);-sin(D1) cos(D1) 0;sin(E)*cos(D1) sin(E)*sin(D1) cos(E)]*[sin(rfy(rr))*cos(rfw(rr));sin(rfy(rr))*sin(rfw(rr));cos(rfy(rr))];
    F2(:,rr)=[cos(E)*cos(D2) cos(E)*sin(D2) -sin(E);-sin(D2) cos(D2) 0;sin(E)*cos(D2) sin(E)*sin(D2) cos(E)]*[sin(rfy(rr))*cos(rfw(rr));sin(rfy(rr))*sin(rfw(rr));cos(rfy(rr))];
    F3(:,rr)=[cos(E)*cos(D3) cos(E)*sin(D3) -sin(E);-sin(D3) cos(D3) 0;sin(E)*cos(D3) sin(E)*sin(D3) cos(E)]*[sin(rfy(rr))*cos(rfw(rr));sin(rfy(rr))*sin(rfw(rr));cos(rfy(rr))];
    F4(:,rr)=[cos(E)*cos(D4) cos(E)*sin(D4) -sin(E);-sin(D4) cos(D4) 0;sin(E)*cos(D4) sin(E)*sin(D4) cos(E)]*[sin(rfy(rr))*cos(rfw(rr));sin(rfy(rr))*sin(rfw(rr));cos(rfy(rr))];
    F5(:,rr)=[cos(E)*cos(D5) cos(E)*sin(D5) -sin(E);-sin(D5) cos(D5) 0;sin(E)*cos(D5) sin(E)*sin(D5) cos(E)]*[sin(rfy(rr))*cos(rfw(rr));sin(rfy(rr))*sin(rfw(rr));cos(rfy(rr))];
end
%%----local theata and fy---------
%���ֲ�ֱ������ϵ(x',y',z')�任Ϊ�ֲ�������
thta1=zeros(1,num_angle);thta2=zeros(1,num_angle);thta3=zeros(1,num_angle);thta4=zeros(1,num_angle);thta5=zeros(1,num_angle);
fai1=zeros(1,num_angle);fai2=zeros(1,num_angle);fai3=zeros(1,num_angle);fai4=zeros(1,num_angle);fai5=zeros(1,num_angle );
for tt=1:num_angle
    thta1(1,tt)=acos(F1(3,tt));       fai1(1,tt)=atan(F1(2,tt)/F1(1,tt));  %signal in 1st subarray   �ź��ڵ�һ��������   acos�����Ƿ����ң�atan�����Ƿ�����
    thta2(1,tt)=acos(F2(3,tt));       fai2(1,tt)=atan(F2(2,tt)/F2(1,tt));
    thta3(1,tt)=acos(F3(3,tt));       fai3(1,tt)=atan(F3(2,tt)/F3(1,tt));
    thta4(1,tt)=acos(F4(3,tt));       fai4(1,tt)=atan(F4(2,tt)/F4(1,tt));
    thta5(1,tt)=acos(F5(3,tt));       fai5(1,tt)=atan(F5(2,tt)/F5(1,tt));
end
%-----polarisation componet of local antenna pattern------�ֲ�����ͼ�ļ���ģʽ

gthta1=sin(thta1-fai1);gfai1=cos(thta1-fai1);  %-----signals to 1st subarray---------�źŵ���һ��������
gthta2=sin(thta2-fai2);gfai2=cos(thta2-fai2);  %-----signals to 2nd subarray---------�źŵ��ڶ���������
gthta3=sin(thta3-fai3);gfai3=cos(thta3-fai3);  %-----signals to 3rd subarray---------�źŵ�������������
gthta4=sin(thta4-fai4);gfai4=cos(thta4-fai4);  %-----signals to 4th subarray---------�źŵ����ĸ�������
gthta5=sin(thta5-fai5);gfai5=cos(thta5-fai5);  %-----signals to 5th subarray---------�źŵ������������

%-----anteena directin pattern matrix------------------���߷���ͼ����
GGthta=[gthta1;gthta2;gthta3;gthta4;gthta5];
GGfai=[gfai1;gfai2;gfai3;gfai4;gfai5];
Gthta=GGthta.';   %.'�������ǰ������GGthta�о���ת��Ϊ�о��󣬼�ȡת��
Gfai=GGfai.';  
GG=Gthta.'*K1+Gfai.'*K2;  %  ����ͼ����sensor pattern of two signals  �����źŵĴ�����ģʽ
%--------------------------------------------------------------------------

AA=kr(Ac,Az);     % steering matrix without sensor pattern ������ģʽ��ת�����
A0=kr(GG.*Ac,Az);   %kr������������ָ����kr�Ӻ���   .*ָ�������������Ӧÿ��Ԫ�����
A00=kr(A0,P1);    %���򼫻��Ƕ�����ʸ��
%%------------------------------------------------
numss=0;
testnum=1:1;
cc=length(testnum);   %����testnum�������������еĽϴ�ֵ��������
anglefw=zeros(cc,num_angle);
anglefy=zeros(cc,num_angle);
for  ii=1:cc;
    
    %------signal source and noise produce-------------�ź�Դ����������
    
    for p=1:num_angle
        for q=1:kp
            p1=rand(1,1);   %rand�����ǲ���1��1�е�һ��1���ڵ������
            p2=rand(1,1);
            sr(p,q)=sqrt(-2*snr1*snr1*log(p1))*cos(2*pi*p2);    %real part     sqrt�����Ǽ���ƽ������
            si(p,q)=sqrt(-2*snr1*snr1*log(p1))*sin(2*pi*p2);    %image part
            s(p,q)=sr(p,q)+j*si(p,q);    
        end
    end
  
    n1=wgn(sm*6,kp,0,'complex');
  x0=A00*s+n1; % no mutual coupling    �޻������
  
d=dix;    %Spacing   ��Ԫ���
T=kp;              %Snapshots   ������
thetatest=[-90:90]; 
%Sampling Grid   ��������
thetanum=length(thetatest);%Sampling Number   ��������    thetanum=181
K=size(theta_fw,2);                   %Source Number    Դ��   size�����Ƿ���theta_fw�ĵ�2ά�ĳߴ�    K=3
M=sm;                %Array Number    ��Ԫ��5
iterNum=10000;      %Iteration Number    ������  
epsilon_min=0.0001;     %Threshold     ��ֵ
gamma=ones(thetanum,1);    %gamma��181��1�е�ȫ1����181*1
Nsample=length(SNR);     %Nsample�������20
estsbl_theta=zeros(Nsample,K);     %estsbl_theta��20��10�е�ȫ0����
estmusic_theta=zeros(Nsample,K);   %estmusic_theta��20��10�е�ȫ0����
theta_error_sbl=zeros(Nsample,1);  %theta_error_sbl��20��1�е�ȫ0����
theta_error_msc=zeros(Nsample,1);  %theta_error_msc��20��1�е�ȫ0����
Ntrial=1;
P=ones(NN,N);    %P��91��181�е�ȫ1����
for nNsample=1:Nsample    %��һ��ѭ��ѭ��20��
    S=SNR(nNsample);
    snr=1;     %Incident Power\\  ���书��
    for n=1:Ntrial     %�ڶ���ѭ��ѭ��1��
        epsilon=1;     %������
        sigma_2=1;      %Noise Variance    ��������
        Pn=sigma_2;    %Noise Power        ��������
        St=zeros(thetanum,T);    %St��181��200�е�ȫ0����
        DD1=A00'*ones(6*sm,sm);     %sm=5,
        D=exp(1j*pi*[0:M-1]'*sin(thetatest*pi/180)); %m=5  ���걸�ֵ�
        source_amplitude = sqrt(sigma_2.*snr)*ones(1,T);                     %Source Amplitude    Դ���   sqrt�����Ǽ���ƽ������
        source_wave = sqrt(0.5)*(randn(T,K) + j*randn(T,K)); %Source Wave    Դ��   randn�����ǲ�����ֵΪ0������ ��^2 = 1����׼��� = 1����̬�ֲ�������������ĺ���
        SignalVector=source_amplitude.*source_wave.';       %Source Matrix   Դ����
        NoiseVector = sqrt(0.5*sigma_2)*(randn(M,T)+j*randn(M,T));  %Noise Matrix    ��������
        %% SBL
        for k=1:K     % k=3������ѭ��ѭ��3��
            for i=1:thetanum     %���Ĳ�ѭ��ѭ��181��
                if thetatest(i)==theta_fw(k)
                    St(i,:)=SignalVector(k,:);      %Sparse Matrix    ϡ�����
                end
            end
        end
        Xt = D*St + NoiseVector;               %Recieved Signal    �����ź�
        %% Iteration    ����
        i=1;
        while (i<iterNum) && (epsilon>epsilon_min)     %�����ǵ�����1000��
            pregamma=gamma;                 %Old gamma    gamma��181��1�е�ȫ1����
            Gamma=diag(pregamma);           %����ԽǾ���181*181
            sigma_x(:,:)=sigma_2.*eye(M,M)+D*Gamma*D';    % M=5eye�����ǵ�λ����������Ҫ�����Խ�Ԫ��Ϊ1������Ԫ��Ϊ0�ľ��󣬴˴�Ϊ����M��M�еľ���
            U_W(:,:)=Gamma*D'*inv(sigma_x)*Xt;            %Update mean of W   181*200 ����W��ֵ    inv�����Ǿ������������Ӧ�þ�����ʹ�ø�����
            sigma_W=Gamma-Gamma*D'*inv(sigma_x)*D*Gamma;   %Update Covariance of W     ����W��Э����
            for k=1:thetanum              %ѭ��ѭ��181��
                u(k)=norm(U_W(k,:),2);    %����2����
                 gamma(k)=u(k)^2/K/(1-sigma_W(k,k)/pregamma(k));   %Update  gamma, i--itteration, k--sampling grid ��������   
%                 gamma(k)=u(k)^2/T+sigma_W(k,k);
%                 J(k)=sigma_W(k,k)/gamma(k);%%%#
                J(k)=sigma_W(k,k)/pregamma(k);%%%#������gamma(k)����pregamma(k)���ɣ�����
            end
            L=sum(J);
            %             sigma_2=sqrt((Xt-D*U_W)*(Xt-D*U_W)')/M/(M-thetanum+L);
            %             epsilon=norm(gamma-pregamma,inf)/norm(pregamma,inf);
            sigma_2=(Xt-D*U_W)*(Xt-D*U_W)'/K/(M-thetanum+L);     %Update Variance of Noise    ������������
            epsilon=norm(gamma-pregamma,inf)/norm(pregamma,inf);    %Update Threshold         ������ֵ    norm�������ĺ���
            i=i+1;
            estsbl_theta(n,:)=peak_seek(gamma,thetatest,K);      %����peak_seek�Ӻ���
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

 