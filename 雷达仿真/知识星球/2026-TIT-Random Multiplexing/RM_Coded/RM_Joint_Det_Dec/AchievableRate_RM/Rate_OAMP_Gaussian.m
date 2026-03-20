function [R_OAMP, R_Sep] = Rate_OAMP_Gaussian(snr,v_LE, rho_LE, phi_1,snr_end)

R_OAMP = zeros(1, length(snr)); 
R_Sep = zeros(1, length(snr)); 

for j=1:length(snr)
    %rate
    R_OAMP(j) = integral(@(x) Min_stransfer(x, v_LE, rho_LE(j,:), phi_1(j)), 0, snr_end(j));
    R_Sep(j) = integral(@(x) Transfer_Sep(x, v_LE, rho_LE(j,:), phi_1(j)), 0,snr_end(j));
end
  
R_OAMP =R_OAMP/log(2);
R_Sep=R_Sep/log(2);
end

%%
function v = Min_stransfer(x, v_LE, rho_LE, phi_1)
%NLE
v_Gaussian = MMSE_Gaussian(x);%解调的线

v_Gau1=v_Gaussian(x<=phi_1);%???
v_Gau2=v_Gaussian(x>phi_1);

%optimal 
v_part2  = Opt_MMSE(x, v_Gau2, rho_LE, v_LE, phi_1);

v=[v_Gau1, v_part2];

% plot(rho_LE, v_LE, '-r');
% hold on;
% plot(x, v, '-b');
end

function v=Transfer_Sep(x,v_LE, rho_LE, phi_1)
%NLE
v_Gaussian = MMSE_Gaussian(x);
v_Gau1=v_Gaussian(x<=phi_1);
v_Gau2=v_Gaussian(x>phi_1);

v_part2=Sep_MMSE(x, v_Gau2, rho_LE, v_LE, phi_1);

v=[v_Gau1, v_part2];

% plot(rho_LE, v_LE, '-r');
% hold on;
% plot(x, v, '-b');
end

%%

%%
function v = MMSE_Gaussian(snr)
%x~CN(0,1)
v=1./(1+snr);
end