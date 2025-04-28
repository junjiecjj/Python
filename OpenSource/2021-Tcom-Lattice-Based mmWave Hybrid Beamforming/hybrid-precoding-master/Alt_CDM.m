% the function of  alternating minimization+CDM proposed by
% J. Chen, ¡°Hybrid beamforming with discrete phase shifters for millimeter-wave massive MIMO systems,¡± IEEE Trans. Vehicular Technology, vol. 66, no. 8, pp. 7604¨C7608, 2017.
                
% Author  : Shanxiang Lyu(shanxianglyu@gmail.com)
% Date    : 2020-June

function [ FRF, FBB ] = Alt_CDM(Fopt,cons,NRF,iter)
 
if nargin <=3
    iter=10;
end

 [Ns,Nt]=size(Fopt');
    %Alternating minimization
    FRF=2*randi([0,1],Nt,NRF)-ones(Nt,NRF);%Initialization
    for p=1:iter
        FBB=pinv(FRF)*Fopt;
        FT=Fopt.';
          for k=1:Nt
            FRT(1:NRF,k)=CDM(FT(:,k),FBB.',cons,6);% the last parameter is the number of full loops;
          end
        FRF=FRT.';
    end
 
     FRF=FRF/sqrt(Nt);
     FBB=pinv(FRF)*Fopt;

end
%   CDM reduced from Gibbs
%   written by Shanxiang Lyu (s.lyu14@imperial.ac.uk)
%   Last updated on Mar. 14 /2017
function [xhat]=CDM(y,H,cons,Iter)
% global sini;
% global sigmaw2;
N=size(H,2);
 
z=ones(N,1);

%sigma2=max(1/(2*pi),sigmaw2);
indRan=0;
    for count=1:Iter*N
        
     % fit(count)=norm(y-H*z);
     % xhat_mat(1:N,count)=z; 
        
       indRan=mod(indRan,N)+1;
       ytil=y-H*z+H(:,indRan)*z(indRan);
       htil=H(:,indRan);
     %  mu=(ytil.'*htil)/(norm(htil)^2);%sampling mean .' or '
      mu=(htil'*ytil)/(norm(htil)^2);%sampling mean .' or '
      
     %  va=sigma2/(norm(htil)^2);%sampling variance
     %  z(indRan)=Samp(mu,va,cons);
   z(indRan)= ClimbOne(mu,cons);
        %z(indRan)= mu;
    end

% ind=find(fit==min(fit));
% xhat=xhat_mat(1:N,ind(1));
xhat=ClimbVec(z,cons);
end

function s_out=ClimbVec(s_in,cons)

[M,N]=size(s_in);

% len=max(M,N);

s_out=s_in;

for i=1:M
    for j=1:N
        
        s_out(i,j)=ClimbOne(s_in(i,j),cons);
        
    end
end
end

function s_out=ClimbOne(s_in,cons)
 
len=size(cons,2);

s_com2=ones(1,len)*s_in;

s_com=abs(cons-s_com2);

ind=find(s_com==min(s_com));%if many distances are equal, we returen the first one

if ~isempty(ind)
s_out=cons(ind(1));
else
 s_out=1;
end

end

