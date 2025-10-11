function [SP]=doa_music(X,L,d,d2)
twpi = 2*pi;
Rxx=X*X';
[EV,D]=eig(Rxx);%%%% 
EVA=diag(D).';
[EVA,I]=sort(EVA);
EV=fliplr(EV(:,I));
derad = pi/180;
En=EV(:,L+1:end);
% MUSIC
    thes1=0:90;
    thes2=0:90;
    num1=0;
    for iang1 = thes1
        num1=num1+1;
        phim1=derad*iang1;
        num2=0;
        for iang2 = thes2
            num2=num2+1;
            phim2=derad*iang2;
            a1=exp(-1i*twpi*d*sin(phim1)*sin(phim2)).';
            a2=exp(-1i*twpi*d2*sin(phim1)*cos(phim2)).';
            a = kron(a2,a1);   
            SP(num1,num2)=1/(a'*En*En'*a);
        end
    end
    SP=abs(SP);
    SPmax=max(SP(:));
    SP=SP/SPmax;
   




