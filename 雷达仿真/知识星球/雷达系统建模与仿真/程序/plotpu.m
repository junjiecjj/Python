function plotpu(x)
%绘出随机数的功率谱密度函数频域的图形。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%y=gaussianpu(x);
%r=correlation(y);
w=fft(x);
w=abs(w);
v=2*pi/length(w);
i=0:v:(2*pi-v);
plot(i,w);