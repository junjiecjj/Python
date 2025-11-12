function y=jilei(x)
%×ö»ýÀÛ
%%%%%%%%%%%%%%%%%%%%%%%%%%%
[M,N]=size(x);
N1=N/10;
y=zeros(M,N1);
for i=1:N1
    for j=1:10
        y(1:M,i)=y(1:M,i)+x(1:M,i*j);
    end
end
w=y(:)';
i=0:length(w)-1;
plot(i,w);