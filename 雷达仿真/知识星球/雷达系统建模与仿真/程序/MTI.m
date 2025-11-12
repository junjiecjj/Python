function [s1,s2]=MTI(x1,x2)
%做动目标检测
%%%%%%%%%%%%%%%%%%%%%%%%%%%
[M,N]=size(x1);
s1=zeros(M,N);
for i=1:M
    for j=1:N
        if(i==1)
           s1(i,j)=x1(i,j); 
        else
           s1(i,j)=x1(i,j)-x1(i-1,j); 
        end
    end
end
for i=1:M
    for j=1:N
        if(i==1)
           s2(i,j)=x2(i,j); 
        else
           s2(i,j)=x2(i,j)-x2(i-1,j); 
        end
    end
end
w1=s1(:)';
w2=s2(:)';
i=0:length(w1)-1;
subplot(2,1,1);
plot(i,w1),title('MTI');
subplot(2,1,2);
plot(i,w2);