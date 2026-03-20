function [R_OAMP, R_Sep]  = Rate_OAMP_QPSK(snr,v_LE, rho_LE, phi_1,snr_end)

R_OAMP = zeros(1, length(snr)); 
R_Sep = zeros(1, length(snr)); 
for j=1:length(snr)
    %rate
    %j
    R_OAMP(j) = integral(@(x) Min_stransfer(x, v_LE, rho_LE(j,:), phi_1(j)), 0,snr_end(j));
    R_Sep(j) = integral(@(x) Transfer_Sep(x, v_LE, rho_LE(j,:), phi_1(j)), 0,snr_end(j));
end
  
R_OAMP =R_OAMP/log(2);
R_Sep=R_Sep/log(2);
end

%%
function v = Min_stransfer(x,v_LE, rho_LE, phi_1)
%NLE
%x
v_QPSK = MMSE_QPSK(x);
v_part1=v_QPSK(x<=phi_1);
v_part2=v_QPSK(x>phi_1);

% optimal 
vpart2 = Opt_MMSE(x, v_part2, rho_LE, v_LE, phi_1);

v=[v_part1, vpart2];

% plot(rho_LE, v_LE, '-r');
% hold on;
% plot(x, v, '-b');

end

function v=Transfer_Sep(x,v_LE, rho_LE, phi_1)
%NLE
v_QPSK = MMSE_QPSK(x);
v_part1=v_QPSK(x<=phi_1);
v_part2=v_QPSK(x>phi_1);

vpart2=Sep_MMSE(x, v_part2, rho_LE, v_LE, phi_1);

v=[v_part1, vpart2];

% plot(rho_LE, v_LE, '-r');
% hold on;
% plot(x, v, '-b');

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