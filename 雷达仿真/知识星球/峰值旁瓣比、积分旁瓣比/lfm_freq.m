T=10e-6;                          %时宽10us
B=30e6;                           %线性调频信号的带宽30MHz
K=B/T;                            %线性调频系数
a=[0.6,0.8,1,1.2,1.4,1.6];        %过采样倍数
Fs=a*B;                           %采样率
Ts=1./Fs;                         %采样间隔
N=T./Ts;
k0=length(a);
for k=1:k0
    t=linspace(-T/2,T/2,N(k));
    St=exp(j*pi*K*t.^2); 
    freq=linspace(-B/2,B/2,N(k));
    figure;
    plot(freq*1e-6,fftshift(abs(fft(St))));
    title(['当采样率为',num2str(a(k)),'B时，线性调频信号的幅频特性']);
    xlabel('Frequency in MHz');
    grid on;axis tight;
end