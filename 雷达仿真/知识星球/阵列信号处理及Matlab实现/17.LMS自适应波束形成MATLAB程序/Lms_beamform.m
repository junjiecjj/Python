% LMS波束形成的MATLAB仿真程序
% Developed by xiaofei zhang (南京航空航天大学 电子工程系 张小飞）
% EMAIL:zhangxiaofei@nuaa.edu.cn, fei_zxf@163.com

clear all
close all
clc
M=16;                                     % the number of antenna
K=2;                                      % the number of sources 
theta=[0 30];                             % DOA
d=0.3;                                    % antenna spacing
N=500;                                    % samples 
Meann=0;  varn=1;                         % mean of noise,variance of noise                               
SNR=20;                                   % signal-to-noise ratio 
INR=20;                                   % interference-to-noise ratio 
pp=zeros(100,N);pp1=zeros(100,N);
rvar1=sqrt(varn) * 10^(SNR/20);           % power of signal 
rvar2=sqrt(varn) * 10^(INR/20);           % power of interference 
for q=1:100 
s=[rvar1*exp(j*2*pi*(50*0.001*[0:N-1]));rvar2*exp(j*2*pi*(100*0.001*[0:N-1]+rand))]; % generate the source signals 
A=exp(-j*2*pi*d*[0:M-1].'*sin(theta*pi/180));   % the direction matrix  
e=sqrt(varn/2)*(randn(M,N)+j*randn(M,N));       % the noise 
Y=A*s+e;                                        % the received data 
 
% LMS algorithm
L=200;
de =s(1, :); 
mu=0.0005; 
w = zeros(M, 1); 
for k = 1:N 
    y(k) = w'*Y(:, k);               % predict next sample and error 
    e(k)  = de(k) - y(k);            % error 
    w = w + mu * Y(:,k)*conj(e(k));  % adapt weight matrix and step size 
end 
end

% beamforming using the LMS method 
beam=zeros(1,L); 
for i = 1 : L 
   a=exp(-j*2*pi*d*[0:M-1].'*sin(-pi/2 + pi*(i-1)/L)); 
   beam(i)=20*log10(abs(w'*a)); 
end 
  
% plotting 
figure 
angle=-90:180/200:(90-180/200); 
plot(angle,beam); grid on
xlabel('方向角/degree'); 
ylabel('幅度响应/dB'); 
figure 
for k = 1:N 
    en(k)=(abs(e(k))).^2; 
end 
semilogy(en); hold on
xlabel('迭代次数'); 
ylabel('MSE');  