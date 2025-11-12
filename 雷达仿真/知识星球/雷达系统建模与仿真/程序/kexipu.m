function y=kexipu(m,n)
%由数据量为n的高斯白噪声产生向量为n，功率谱为柯西谱的高斯随机向量
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
wc=2*pi*256;
T0=1/(256*m);
x=gaussian(n);
y=zeros(1,n);
y(1)=wc*T0*x(1);
for i=2:n
    y(i)=wc*T0*x(i)+exp(-wc*T0)*y(i-1);
end
b=canshu(y);
%y=y-b(1);　　　　　 %去掉直流分量　　　
y=conv(y,y);
y=fft(y);
y=abs(y);
i=1:2*n-1;
plot(i,y)