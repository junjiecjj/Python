function [ v_final, rho_final] = OAMP_SEtransfer(M,Nu,beta, kappa, snr)

%%
load('GG_m_0.mat');
[U, Dia, V] = svd(GG_m_0);
dia   = diag(Dia);
dia=dia.';
% Nr = fix( Nu./beta ); 
% dia = zeros(length(beta),Nu);
% T = min(Nr,Nu); 
% for i=1:length(beta)
%     dia(i,1:T(i)) = kappa.^(-[0:T(i)-1] / T(i));
%     dia(i,:) = sqrt(Nu) * dia(i,:) / norm(dia(i,:));      % normalized Tr{A'A}/Nu=1
% end

%% Achievable Rate of OAMP
%  C = C_Gau(dia,snr)
%  fprintf('Gaussian Capacity = %g\n', C);
 
%% OAMP SE
[N_beta,~] = size(dia);
Omega_OAMP = zeros(N_beta, length(snr)); 
step = snr/40;                   %% require to modifed the number of step to statisfy the match condition
rho =0:step:snr;
v_part1=[];
rho_part1=[];
v_part2=[];
rho_part2=[];

for i=1:N_beta
    ind_snr= 1;
    vtmp=[];
    flag_star =0;
    Omega_S= [];
    for snr_j =0:step:snr
        [Omega_OAMP(i,ind_snr), vpart1, vpart2, v_QPSK] = Min_stransfer(snr_j, M,Nu, dia(i,:), snr^-1);
%         Omega_S=[Omega_S, v_QPSK];
%         if length(vpart1)>0 && length(vpart2)==0 
%             v_part1 = [v_part1, vpart1];
%             rho_part1 = [rho_part1, snr_j]; 
%         end
%         if length(vpart1)==0 && length(vpart2)>0 
%             v_part2 = [v_part2, vpart2];
%             rho_part2 = [rho_part2, snr_j];
%         end
%         if flag_star==0 && length(vpart1)==0 && length(vpart2)>0
%             if (ind_snr-1)~=0
%                 rho_star =  rho_part1(ind_snr-1);
%                 v_star = v_part1(end);
%                 flag_star = 1;
%             else
%                 rho_star =  rho_part2(0);
%                 v_star = v_part2(0);
%                 flag_star = 1;               
%             end       
%         end
         ind_snr = ind_snr+1;
    end
end

v_final = Omega_OAMP(1,:);
rho_final = rho;
end

%% functions
function C = C_Gau(dia, snr)
    
[N_beta, Nu] = size(dia);
N_snr = length(snr);
C = zeros(N_beta, N_snr);
for i = 1:N_beta
   for j=1:N_snr
        C(i,j) = 1/Nu * sum( log( 1 + snr(j) * dia(i,:).^2 ) );
   end
end
C = C/log(2);
end

%%
function [v, v_part1, v_part2, v_QPSK] = Min_stransfer(x,M, Nu, dia, Sigma_n)

%phi_1 = Phi_Transfer(1, Nu, dia.^2, Sigma_n);
phi_1= LE_OAMP_SE(1, dia,  Sigma_n, M, Nu);
v_QPSK = MMSE_QPSK(x);
v_part1 = v_QPSK(x<=phi_1);
v_part2 = v_QPSK(x>phi_1);
v_part2 = min( LMMSE1(x(x>phi_1),M, Nu, dia, Sigma_n), v_part2 );
v  = [v_part1 v_part2];
end

%%
function v_out = LMMSE(rho, Nu, dia, Sigma_n)
precious = 10^-5;    
v1 = 10^-20*ones(size(rho));
v2 = ones(size(rho));
dia2=dia.^2;
while (norm(v1-v2)>precious)
    v = (v1+v2)/2;
    t = Phi_Transfer(v, Nu, dia2, Sigma_n)- rho;
    v1(t>0) = v(t>0);  
    v2(t<=0) = v(t<=0);  
end
v_out = 1 ./ (rho + 1./v2);

end

%%
function rho = Phi_Transfer(v, Nu, dia2, Sigma_n)
% v and rho are 1xK
% dia is row 1xNu
rho =  v.^-1 .* ( Nu./sum(1./( Sigma_n^-1 * dia2' * v+ 1) ) -1) ;
end

%%
function v = MMSE_QPSK(snr)
%snr = snr/2;   % scale from BPSK for QPSK
max=100;
v = zeros(size(snr));
for i=1:length(snr)
    v(i) = 1 - integral(@(x) f_QPSK(x,snr(i)), -max, max);
end

%% 
function y = f_QPSK(x,snr) 
y  = exp(-x.^2/2)/sqrt(2*pi) .* tanh(snr - sqrt(snr).*x);
end
end

function  v_post= LE_OAMP_SE(v, dia, v_n, M, N)
v_post=zeros(size(v,1),size(v,2));
    for i=1:length(v)
    rho = v_n / v(i);
    Dia = [dia.^2; zeros(N-M, 1)];
    D = 1 ./ (rho + Dia);
    v_post(i) = v_n / N * sum(D);
    end
end