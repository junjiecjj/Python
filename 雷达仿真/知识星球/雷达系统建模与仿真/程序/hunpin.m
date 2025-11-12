function y=hunpin(s,f0,fs,f1)
%%%%%%%%%%%%%%%%%%%%%%%%%%进行混频，输出为中频信号
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[M,N]=size(s);
i=0:M-1;
z1=cos(2*pi*2*f0*i/3/fs);
y1=zeros(M,N);
for i=1:N
    y1(1:M,i)=(z1.*s(1:M,i)')';
end
w1=(f0-5*f1)*2*pi/(3*fs);%w1=pi/9
w3=(f0-2*f1)*2*pi/(3*fs);%w3=1.6pi/9
w2=(f0+8*f1)*2*pi/(3*fs);%w2=0.4pi
w4=(f0+5*f1)*2*pi/(3*fs);%w4=pi/3
th=min((w3-w1),(w2-w4)); %w3-w1=0.2pi/3,w2-w4=0.2pi/3
M1=ceil(6.6*pi/th)+1;    %M1=99
%n=[0:M1-1];
w5=(w1+w3)/2;w6=(w4+w2)/2;%w5=1.3pi/9,w6=1.1pi/3
h=wide(w6,M1)-wide(w5,M1);
wth=(hamming(M1))';
h=h.*wth;
L=length(h);
z=zeros(M+L-1,N);
y=zeros(M,N);
K=ceil(L/2);
for i=1:N
    for j=1:10
        z(1+(j-1)*M/10:j*M/10+L-1,i)=conv(h,s(1+(j-1)*M/10:j*M/10,i)')';
        y(1+(j-1)*M/10:j*M/10,i)=z(1+(j-1)*M/10+K:j*M/10+K,i);
    end
end
w=y(:)';
v=y1(:)';
subplot(4,1,1);
i=0:length(w)-1;
plot(i,w),title('混频');
subplot(4,1,2);
plotpu(v),title('频谱');
subplot(4,1,3);
plotpu(w);
subplot(4,1,4);
plotpu(h),title('幅频特性');   
    
