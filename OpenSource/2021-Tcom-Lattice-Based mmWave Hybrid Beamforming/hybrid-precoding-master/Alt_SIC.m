% lattice-based alternating minimization+Babai(SIC) method
% ref: Shanxiang Lyu, Zheng Wang, Zhen Gao, Hongliang He, Lajos Hanzo, "Lattice-Based mmWave Hybrid Beamforming", IEEE Transactions on Communications,2021
% Author  : Shanxiang Lyu (shanxianglyu@gmail.com, https://sites.google.com/view/shanx)
% Date    : 2020-June

function [ FRF, FBB ] = Alt_SIC(Fopt,cons,NRF,iter)
 
if nargin <=3
    iter=10;
end

 [~,Nt]=size(Fopt');
    %Alternating minimization
    FRF=2*randi([0,1],Nt,NRF)-ones(Nt,NRF);%Initialization
    for p=1:iter
        FBB=pinv(FRF)*Fopt;
        FT=Fopt.';
          for k=1:Nt
           FRT(1:NRF,k)=SIC(FT(:,k),FBB.',cons); 
          end
        FRF=FRT.';
    end
     FRF=FRF/sqrt(Nt);
     FBB=pinv(FRF)*Fopt;
end

function [x_hat]=SIC(yy,H,cons)

    [M,N]=size(H);
    [Q,R2]=qr(H); 

    if M>N
        Q1=Q(1:M,1:N);
        y=Q1'*yy;
    else
        y=Q'*yy;
    end
    R=R2(1:N,1:N); 

    x_hat=zeros(N,1);
    for i=N:-1:1
        x_hat(i)=ClimbOne(y(i)/R(i,i),cons);
        if i>=2
        y=y-R(:,i)*x_hat(i);
        end
    end
end

function s_out=ClimbOne(s_in,cons)

len=size(cons,2);

s_com2=ones(1,len)*s_in;

s_com=abs(cons-s_com2);

ind=find(s_com==min(s_com));

if ~isempty(ind)
    s_out=cons(ind(1));
else
    s_out=1;
end
end


