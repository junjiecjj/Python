function y=duishuzhengtaipu(a,b,n)
%由数据量为n的高斯白噪声产生向量为n，功率谱为高斯型的对数正态随机向量
%a表示标准方差,b表示均值
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
z1=gaussian(n);
x=gaussianpu(z1);
y=a*x;
y=exp(y);
y=b*y;
b=canshu(y);
y=y-b(1);   %去掉直流分量
%y=conv(y,y);
%y=fft(y);
%y=abs(y);
%i=1:2*n-1;
%plot(i,y)
%plotpu(y)