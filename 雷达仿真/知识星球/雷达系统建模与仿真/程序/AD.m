function [x,y]=AD(z1,z2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Ä£Êý×ª»»
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m1=min(min(z1));
m2=min(min(z2));
Vmax=6;
N=12;
x=Vmax/(2^N)*floor((z1-m1)*2^N/Vmax);
y=Vmax/(2^N)*floor((z2-m2)*2^N/Vmax);
[M,N]=size(z1);
for i=1:N
    x(1:M,i)=Vmax/(2^N)*floor((z1(1:M,i)-m1)*2^N/Vmax);
    y(1:M,i)=Vmax/(2^N)*floor((z2(1:M,i)-m2)*2^N/Vmax);
end
w1=x(:)';
w2=y(:)';
subplot(2,1,1);
plot(w1);
subplot(2,1,2);
plot(w2);