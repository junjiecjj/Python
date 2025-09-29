function [ R] = R_ambity(Fr,R_am )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

Tr=1./Fr;

eps=200;%误差
c=3e8;
R_T=c*Tr/2;%最大不模糊距离。R_am为三种频率测得的模糊距离。

num_all=20;

R_r=zeros(num_all,length(Fr));

for num=1:num_all
    for k=1:length(Fr)
       R_r(num,k)=num*R_T(k)+R_am(k);
    end
end

m=1;
flag=0;
    while m<=num_all&&flag==0
        k=1;
        while k<=num_all&&flag==0
            if abs(R_r(m,1)-R_r(k,2))<eps
               r1=R_r(m,1);
               r2=R_r(k,2);
               l=1;
               while l<=num_all&&flag==0
                   
                   if abs(r1-R_r(l,3))<eps
                       r3=R_r(l,3);
                       flag=1;
                       
                   end
                      l=l+1;
               end
               
            end
            k=k+1;
        end
        m=m+1;
    end
 R=(r1+r2+r3)/3;       
            

end

