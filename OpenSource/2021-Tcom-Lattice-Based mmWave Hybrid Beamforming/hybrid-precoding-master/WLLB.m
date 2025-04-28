% the function of finite resolution PSs based precoding proposed by
% Z. Wang, M. Li, Q. Liu, and A. L. Swindlehurst, ¡°Hybrid precoder and combiner design with low-resolution phase shifters in mmwave MIMO systems,¡± J. Sel. Topics Signal Processing, vol. 12, no. 2, pp. 256¨C269, 2018
               
% Download Link : http://ice.dlut.edu.cn/LiMing/doc/Hybrid_precoder_and_combiner_design_with_low_resolution_phase_shifters.rar

%%% Input: Initial analog precoder and combiner, channel matrix,
%%% number of transmit antennas, number of receive antennas, number of RF
%%% chains at both sides, number of data streams.
%%% Output: Low resolution hybrid precoder and combiner


function [F, W] =WLLB(Frf_temp,Wrf_temp,H,Nt,Nr,Nrf,Ns,b,iter)

if nargin <=8
    iter=10;
end
 
omiga=exp(1j*2*pi/(2^b));
f_ind=[0:1:2^b-1]';
F_set=omiga.^(f_ind); 


Q=H;
[U S V]=svd(H);
U=U(:,1:Nrf);
V=V(:,1:Nrf);
S=S(1:Nrf,1:Nrf);

ff=zeros(Nt,Nrf);
ww=zeros(Nr,Nrf);

Frf=zeros(Nt,Nrf);
Wrf=zeros(Nr,Nrf);

Mat_f=zeros(Nt,size(F_set,1));
Mat_w=zeros(Nr,size(F_set,1));


sigma=1;
ncount=0;
while sigma >0.01 && ncount<=iter
    ncount=ncount+1;
    for k=1:Nrf
        ff(:,k)=[];
        ww(:,k)=[];
        Q=U*inv(0.01*eye(Nrf)+V'*ff*ww'*U)*V';      
        for j=1:Nr
            for ii=1:size(F_set,1)
                Mat_w(:,ii) = Wrf_temp(:,k);
                Mat_w(j,ii)=F_set(ii);
            end
            Product = Mat_w'*Q* Frf_temp(:,k);
            [~, position] = max(Product);
            Wrf_temp(:,k) = Mat_w(:,position);
        end
        
        for i=1:Nt
            for ii=1:size(F_set,1)
                Mat_f(:,ii) = Frf_temp(:,k);
                Mat_f(i,ii)=F_set(ii);
            end
            Product = Wrf_temp(:,k)'*Q*Mat_f;
            [~, position] = max(Product);
            Frf_temp(:,k) = Mat_f(:,position);
        end
        ff=Frf_temp;
        ww=Wrf_temp;
    end
    sigma1=norm(Frf-Frf_temp,'fro');
    sigma2=norm(Wrf-Wrf_temp,'fro');
    sigma=max(sigma1,sigma2);
    Frf=Frf_temp;
    Wrf=Wrf_temp;
end

Heff=Wrf'*H*Frf;
[Ue SIGMA Ve] =svd(Heff);
Fd = Ve(:,1:Ns);
Wd = Ue(:,1:Ns);
 
for i=1:1:Nrf
    Fd(:,i)=Fd(:,i)/sqrt(trace((Frf*Fd(:,i))'*(Frf*Fd(:,i))));
end
F=Frf*Fd;

for i=1:1:Nrf
    Wd(:,i)=Wd(:,i)/sqrt(trace((Wrf*Wd(:,i))'*(Wrf*Wd(:,i))));
end
W=Wrf*Wd;






