function y=swerling2pu(n)
%由数据量为n的高斯白噪声产生向量为n，功率谱为高斯型的斯维凌II型随机向量
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
r=6;
[z1,z2]=gaussian(n);
x1=gaussianpu(z1);
x2=gaussianpu(z2);
y=x1.^2+x2.^2;
y=r*y;
b=canshu(y);
y=y-b(1);     %去掉直流分量
%y=conv(y,y);
%y=fft(y);
%y=abs(y);
%i=1:2*n-1;
%plot(i,y)
%plotpu(y)