function y=gaofang(s,f0,fs,f1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%高频放大器
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[M,N]=size(s);
Vgain=3;                                         %高放增益
w1=(f0-4*f1/3)*2*pi/fs; %w1=26pi/45
w3=(f0-f1/3)*2*pi/fs;   %w3=29pi/45
w2=(f0+7*f1/3)*2*pi/fs; %w2=37pi/45
w4=(f0+4*f1/3)*2*pi/fs; %w4=34pi/45
th=min((w3-w1),(w2-w4));%w3-w1=pi/15,w2-w4=pi/15
M1=ceil(6.6*pi/th)+1;   %M1=99
%n=[0:M1-1];
w5=(w1+w3)/2;w6=(w4+w2)/2;%w5=11pi/18,w6=71pi/90
h=wide(w6,M1)-wide(w5,M1);
w=(hamming(M1))';
h=h.*w;
L=length(h);
s=Vgain*s;
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
subplot(3,1,1);
i=0:length(w)-1;
plot(i,w),title('高放');
subplot(3,1,2);
plotpu(w),title('频谱');
subplot(3,1,3);
plotpu(h),title('幅频特性');