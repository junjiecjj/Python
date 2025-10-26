
clc
clear
close all

%% Hfm
fc=12.5e3;
fs=5*fc;
ts=1/fs;
N=450;
T=(1/fs)*N;

B=6000;
fmax=fc/(1-B/(2*fc));
fmin=fc/(1+B/(2*fc));
bw=fmax-fmin;
t0=fc*N*ts/B;
K=N*ts*fmax*fmin/B;
W=B*N*ts;
t=(-(N-1)/2:1:(N-1)/2).'*ts;
x0=exp(-1i*(2*pi*K*log(1-t/t0)));
m=bw/T;
v=9;
c=1500;
D=1+2*v/c;
ft=fc./(1-(m/fc).*t);
ftr=fc*D./(1-(m/fc).*D*t);

figure(1)
plot(t,ft,t,ftr);
grid on
tr=((-(N-1)/2:1:(N-1)/2).'+50)*ts;
xr=exp(-1i*(2*pi*K*log(1-tr/t0)));

prf=100;
L=round((1/prf)/ts);

x=zeros(L,1);
x(1:1:N,1)=x0;
[afmag,delay,doppler]=ambgfun(x,fs,prf);
ambgu=afmag.*(afmag>0.5);
figure(2)
% contour(delay,doppler,afmag,'ShowText','on');
mesh(delay,doppler,afmag);
xlim([-8e-3,8e-3]);
ylim([-6600,6600]);
xlabel('Delay (second)');
ylabel('Doppler Shift (hertz)');
colormap jet;
colorbar;
grid on

figure(3)
contour(delay,doppler,ambgu);
xlim([-8e-3,8e-3]);
ylim([-7600,7600]);
xlabel('Delay (second)');
ylabel('Doppler Shift (hertz)');
colormap jet;
colorbar
grid on

%%
afmag_T0=afmag(1025,:);
afmag_f0=afmag(:,625);

figure(4)
plot(delay*1000,afmag_T0);
xlabel('Delay (ms)');
ylabel('Amplitude');
grid on
xlim([-2 2]);
ylim([0 1]);
hold on
hline=refline(0,0.7);
hline.Color='r';
hline.LineStyle='--';
hline.LineWidth=1.2;
plot([-0.44/bw -0.44/bw]*1000,[0 1],'r--','LineWidth',1.2);

figure(5)
plot(doppler,afmag_f0);
xlabel('Doppler Shift (hertz)');
ylabel('Amplitude');
xlim([-3/T 3/T]);
ylim([0 1]);
grid on
hold on
hline=refline(0,0.7);
hline.Color='r';
hline.LineStyle='--';
hline.LineWidth=1.2;
plot([-0.44/T -0.44/T]*1000,[0 1],'r--','LineWidth',1.2);

%%signal
tx=(0:1:L-1)*ts*1000;
figure(6)
subplot(2,1,1)
plot(t,real(x0));
ylim([-1.1 1.1]);
xlabel('Time/ms');
ylabel('Amplitude');
grid on

subplot(2,1,2)
nfft=2^nextpow2(450);
fre=(0:1:nfft/2)*fs/nfft/1000;
fftr=fft(x(1:450,1),nfft);
plot(fre,abs(fftr(1:1:nfft/2+1,1)));
xlabel('Frequency/kHz');
ylabel('Amplitude');
grid on
