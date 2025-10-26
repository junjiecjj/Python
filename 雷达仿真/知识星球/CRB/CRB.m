clc
clear
index=1;
m=8; % 阵元数
p=1; % 信号数
N=1024; % 快拍数
d=p;

st1=10; % 俯仰角1
% st2=85; % 俯仰角2

st=[st1];

degrad=pi/180;
snrdata=-10 : 1 : 10;
crb=zeros(1,length(snrdata));
%构建方向矩阵
tmp =-j*pi*sin(st*degrad);
tmp2=[0: m-1]';
a2=tmp2*tmp;
A=exp(a2);

sigma_s = 1.0*ones(1,d);   
S= sqrt(diag(sigma_s/2))*(randn(d,N) + 1j*randn(d,N)); 	% 生成信号
sigma=2;
Ps=S*S'/N;
ps=diag(Ps);
	
% for snap=1:15
for snr=snrdata(1) : 1 : snrdata(length(snrdata))
	refp=2*10.^(snr/10);
	tmp=sqrt(refp./ps);
	SS=diag(tmp)*S;

	Ps=SS*SS'/N;
	R=A*Ps*A'+sigma*eye(m);
	P=Ps*A'*inv(R)*A*Ps;
	D=-degrad*i*pi*cos(st*degrad)*tmp2.*A;
	H=D'*(eye(m,m)-A*inv(A'*A)*A')*D;
	q=sqrt(sigma/(2*N)*inv(real(H.*P)));
	crb(index)=q;
	index=index+1;
end
% 画图
snr=snrdata(1): 1: snrdata(length(snrdata));
plot(snr,crb(1:length(snr)), '--rd',  'LineWidth', 1.2, 'MarkerSize', 8);
legend('CRB'); grid on; 
% set(gca, 'XLim', [snr(1), snr(length(snr))]);                 
% set(gca, 'XTick', snr);        
% set(gca, 'YLim', [0, 1]);  
xlabel('GSNR /dB'); 
ylabel('CRB /degree');
title('Cramer-Rao (Low) bound');
