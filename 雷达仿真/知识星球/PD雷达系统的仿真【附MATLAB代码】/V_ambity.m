function [V] =  V_ambity( Fr,V_am )
c=3e8;
Fc=1e9;% 中心频率【Hz】
Wavelength=c/Fc;                 % 工作波长【m】

V_T=Fr*(Wavelength/2);  %最大不模糊速度，V_am为三种频率测得的模糊速度。         

eps=50;%误差

num_all=100;

V_r=zeros(num_all,length(Fr));

for num=1:num_all
    for k=1:length(Fr)
       V_r(num,k)=num*V_T(k)+V_am(k);
    end
end

m=1;
flag=0;
    while m<=num_all&&flag==0
        k=1;
        while k<=num_all&&flag==0
            if abs(V_r(m,1)-V_r(k,2))<eps
               V1=V_r(m,1);
               V2=V_r(k,2);
               l=1;
               while l<=num_all&&flag==0
                   
                   if abs(V1-V_r(l,3))<eps
                       V3=V_r(l,3);
                       flag=1;
%                    else
%                        printf('wrong ')
                   end
                      l=l+1;
               end
               
            end
            k=k+1;
        end
        m=m+1;
    end
 V=(V1+V2+V3)/3; 


end

