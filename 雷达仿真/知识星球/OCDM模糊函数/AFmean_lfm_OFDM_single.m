function [ AF_u ] = AFmean_lfm_OFDM_single(  N_c,T_b,mu,x,y,weight )
%% lfm_ofdm������ģ��������ֵ
%% ���Բ���
% ϵͳ���ز���
% N_c=12;
% �������ʱ��
% T_b=0.8e-6;
% ��Ƶб��
% mu=20/T_b^2;
% ��һ��ʱ��
% x=-1:0.01:1;
% ��һ��Ƶ��
% y=-10:0.01:10;
% ��������
% weight=ones(1,N_c);
% weight=chebwin(N_c,90);

%% ��������
[X,Y]=meshgrid(x,y);
% ���ز�Ƶ�ʼ��
delta_f=1/T_b;
% ���ز�Ƶ��
f=(0:N_c-1)./T_b;  
% ��������
E=T_b*sum(abs(weight));
[amf,amt]=size(X);
c1=zeros(amf,amt);
a1=T_b-abs(X*T_b);
b1=sinc((delta_f*Y+mu*X*T_b).*a1);
d1=exp(1j*pi*(delta_f*Y+mu*X*T_b).*(T_b+X*T_b));
for k=1:N_c
    temp=exp(1j*2*pi*f(k)*X*T_b-1j*pi*mu*(X*T_b).^2);
    c1=c1+weight(k)*temp;
end
AF_u=a1.*b1.*c1.*d1;
AF_u=AF_u/E;
AF_u=abs(AF_u);


% ģ������ͼ
close all;
figure;grid on;hold on
surf(X,Y,AF_u,'EdgeColor','none')
title('lfm-OFDMģ������','fontsize',14)
xlabel('��һ��ʱ��','fontsize',14),ylabel('��һ��Ƶ��','fontsize',14)
colormap('default')
% surface(x, [0,0], [zeros(1,amt);AF_u(1,:)],'EdgeColor',[0 0 0],'FaceColor',[0 0 0]);
% zlim([0,1]);view(0,90)

end




