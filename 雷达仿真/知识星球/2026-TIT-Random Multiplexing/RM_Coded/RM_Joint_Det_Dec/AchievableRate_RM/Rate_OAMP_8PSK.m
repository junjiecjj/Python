function [R_OAMP, R_Sep]  = Rate_OAMP_8PSK(snr,v_LE, rho_LE, phi_1,snr_end)

R_OAMP = zeros(1, length(snr));
R_Sep = zeros(1, length(snr)); 
for j=1:length(snr)
    %rate
    R_OAMP(j) = integral(@(x) Min_stransfer(x, v_LE, rho_LE(j,:), phi_1(j)),0,snr_end(j));
    R_Sep(j) = integral(@(x) Transfer_Sep(x, v_LE, rho_LE(j,:), phi_1(j)),0,snr_end(j));
end

R_OAMP =R_OAMP/log(2);
R_Sep=R_Sep/log(2);
end

%%
function v = Min_stransfer(x,v_LE, rho_LE, phi_1)
%NLE
v_8PSK = MMSE_8PSK(x);

v_part1=v_8PSK(x<=phi_1);
v_part2=v_8PSK(x>phi_1);
 
% optimal 
vpart2  = Opt_MMSE(x, v_part2, rho_LE, v_LE, phi_1);

v=[v_part1, vpart2];

% plot(rho_LE, v_LE, '-r');
% hold on;
% plot(x, v, '-b');

end

function v=Transfer_Sep(x,v_LE, rho_LE, phi_1)
%NLE
v_8PSK = MMSE_8PSK(x);
v_part1=v_8PSK(x<=phi_1);
v_part2=v_8PSK(x>phi_1);

vpart2=Sep_MMSE(x, v_part2, rho_LE, v_LE, phi_1);

v=[v_part1, vpart2];

% plot(rho_LE, v_LE, '-r');
% hold on;
% plot(x, v, '-b');

end
%%
function v = MMSE_8PSK(snr)

S_8PSK = PSK_Constellation(8);
P_8PSK = ones(size(S_8PSK))/length(S_8PSK);
v = MMSE_Constellation(S_8PSK, P_8PSK, snr);

end