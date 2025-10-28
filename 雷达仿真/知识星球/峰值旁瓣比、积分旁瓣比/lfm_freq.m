T=10e-6;                          %ʱ��10us
B=30e6;                           %���Ե�Ƶ�źŵĴ���30MHz
K=B/T;                            %���Ե�Ƶϵ��
a=[0.6,0.8,1,1.2,1.4,1.6];        %����������
Fs=a*B;                           %������
Ts=1./Fs;                         %�������
N=T./Ts;
k0=length(a);
for k=1:k0
    t=linspace(-T/2,T/2,N(k));
    St=exp(j*pi*K*t.^2); 
    freq=linspace(-B/2,B/2,N(k));
    figure;
    plot(freq*1e-6,fftshift(abs(fft(St))));
    title(['��������Ϊ',num2str(a(k)),'Bʱ�����Ե�Ƶ�źŵķ�Ƶ����']);
    xlabel('Frequency in MHz');
    grid on;axis tight;
end