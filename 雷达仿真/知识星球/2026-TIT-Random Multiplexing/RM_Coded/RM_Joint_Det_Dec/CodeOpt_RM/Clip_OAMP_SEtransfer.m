function [var, rho] = Clip_OAMP_SEtransfer(snr_end,v_LE, rho_LE, phi_1)

step = snr_end/30;                   %% require to modifed the number of step to statisfy the match condition
rho =0:step:snr_end;
var =Min_stransfer(rho, v_LE, rho_LE, phi_1);

end


function v = Min_stransfer(x,v_LE, rho_LE, phi_1)
%NLE
v_QPSK = MMSE_QPSK(x);
v_part1=v_QPSK(x<=phi_1);
v_part2=v_QPSK(x>phi_1);

% optimal 
vpart2 = Opt_MMSE(x, v_part2, rho_LE, v_LE, phi_1);

v=[v_part1, vpart2];

end

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

