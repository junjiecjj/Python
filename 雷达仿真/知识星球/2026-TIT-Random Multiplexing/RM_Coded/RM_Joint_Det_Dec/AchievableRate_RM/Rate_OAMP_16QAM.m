function [R_OAMP, R_Sep]  = Rate_OAMP_16QAM(snr,v_LE, rho_LE, phi_1,snr_end)

R_OAMP = zeros(1, length(snr)); 
R_Sep = zeros(1, length(snr)); 
for j=1:length(snr)
    %rate
    R_OAMP(j) = integral(@(x) Min_stransfer(x, v_LE, rho_LE(j,:), phi_1(j)), 0,snr_end(j));
    R_Sep(j) = integral(@(x) Transfer_Sep(x, v_LE, rho_LE(j,:), phi_1(j)), 0,snr_end(j));
end
  
R_OAMP =R_OAMP/log(2);
R_Sep=R_Sep/log(2);
end

%%
function v = Min_stransfer(x,v_LE, rho_LE, phi_1)
%NLE
v_16QAM= MMSE_16QAM(x);

v_part1=v_16QAM(x<=phi_1);
v_part2=v_16QAM(x>phi_1);

vpart2  = Opt_MMSE(x, v_part2, rho_LE, v_LE, phi_1);

v=[v_part1, vpart2];

% plot(rho_LE, v_LE, '-r');
% hold on;
% plot(x, v, '-b');
end

%%
function v=Transfer_Sep(x,v_LE, rho_LE, phi_1)
%NLE
v_16QAM= MMSE_16QAM(x);
v_part1=v_16QAM(x<=phi_1);
v_part2=v_16QAM(x>phi_1);

vpart2=Sep_MMSE(x, v_part2, rho_LE, v_LE, phi_1);

v=[v_part1, vpart2];
% plot(rho_LE, v_LE, '-r');
% hold on;
% plot(x, v, '-b');

end

%%
function v = MMSE_16QAM(snr)
[S_4PAM,~]= PAM_QAM_Constellation(4);
P_4PAM = ones(size(S_4PAM))/length(S_4PAM);
v = MMSE_PAM(S_4PAM, P_4PAM, snr); 
end