function  AF_m1 = AFmean_single_symbol( N_c,T_b,x,y,d)
%% ����OFDM���ŵ�ƽ��ģ������
% ����ֵ

% % ϵͳ���ز���
% N_c=13;
% % �������ʱ��
% T_b=0.8e-6;
% % ��һ��ʱ��
% x=0:1:1;
% % ��һ��Ƶ��
% y=-8:0.001:8;
[X,Y]=meshgrid(x,y);


% ���ز�Ƶ�ʼ��
delta_f=1/T_b;
% ���ز�Ƶ��
f=(0:N_c-1)./T_b;  
% ��������
E=T_b*sum(d);

% ������ģ��������ֵ
a1=T_b-abs(X*T_b);
b1=sinc(delta_f*Y.*a1);
c1=0;
d1=exp(1j*pi*delta_f*Y.*(T_b+X*T_b));
for k=1:N_c
    temp=exp(1j*2*pi*f(k)*X*T_b);
    c1=c1+d(k)*temp;
end
AF_m0=a1.*b1.*c1.*d1;
AF_m0=AF_m0/E;
AF_m1=abs(AF_m0);


end

