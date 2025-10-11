%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
radeg = 180/pi;
derad=1/radeg;
twpi = 2*pi;
kelm = 9;               % 
dd = 0.5;               % 
d=0:dd:(kelm-1)*dd;     % 
iwave = 3;              % number of DOA
theta = [10 20 30];     % DOA
pw= [1.0  1.0  1];      % power
cb = [ 1 0 0
       0 1 0
       0 0 1];          % source relation 

nv=ones(1,kelm);        % normalized noise variance
n = 200                % 
A=exp(-j*twpi*d.'*sin(theta*derad));%%%% direction matrix
for iter=1:3
S=randn(iwave,n);
snr0=10:3:100
for isnr=1:1
X0=A*S;
X=awgn(X0,snr0(isnr),'measured');
Rxx=X*X'/n;
[EV,D]=eig(Rxx);
EVA=diag(D)'; [EVA,I]=sort(EVA);
EVA=fliplr(EVA); EV=fliplr(EV(:,I));

% LS-unitary-ESPRIT            
estimates=unitary_esprit(dd,Rxx,iwave,1);
doaes(isnr,:,iter)=sort(estimates(1,:))
% TLS-unitary-ESPRIT    
estimates=unitary_esprit(dd,Rxx,iwave,2);
doaes2(isnr,:,iter)=sort(estimates(1,:))

end
end

 
