%% test one specific case and plot power spectrum and frequency response

clear;
clc;
N = 100;
sigma2 = 0.1; % noise power
sigma2jammer = 100; % jammer power
beta = 1;
rho = 1;
s0 = golomb(N);
%s0 = exp(1i*2*pi*rand(N,1));
jammertype = 'spot';

switch jammertype
    case 'white'
        Gamma = (sigma2 + sigma2jammer) * eye(N);
    case 'spot'
        f1 = 0.2;
        Num = 2*N-1;
        power = zeros(Num,1);
        power(floor(f1*Num)+1) = 1;
        r = ifft(power); % [r(0) r(1) ... r(N-1) r*(N-1) ... r*(1)].'
        r = r/abs(r(1)) * sigma2jammer;
        Gamma = toeplitz(r(1:N), [r(1) r(2*N-1:-1:N+1).']) ...
            + sigma2 * eye(N);
    case 'barrage'
        f1 = 0.2; f2 = 0.3;
        Num = 2*N-1;
        power = zeros(Num,1);
        power(floor(f1*Num)+1 : floor(f2*Num)) = 1;
        r = ifft(power); % [r(0) r(1) ... r(N-1) r*(N-1) ... r*(1)].'
        r = r/abs(r(1)) * sigma2jammer;
        Gamma = toeplitz(r(1:N), [r(1) r(2*N-1:-1:N+1).']) ...
            + sigma2 * eye(N);
    otherwise
        error('Unrecognized jammer type');
end

% sCAN = cansisopar(rho, N, s0);
% show_criterion(sCAN, Gamma, beta, 'CAN');
% % 
% sGRA = grad_direct(N, Gamma, beta, s0);
% show_criterion(sGRA, Gamma, beta, 'GRA');
%
% sCYC = cyc_direct(N, Gamma, beta, rho, s0);
% show_criterion(sCYC, Gamma, beta, 'CYC');

[sFRE wFRE mseFRE] = cyc_freq(N, Gamma, beta, rho, s0);
show_criterion(sFRE, Gamma, beta, 'FRE');

% [sMAT wMAT] = cyc_freq_mat(N, Gamma, beta, rho, s0);
% show_criterion(sMAT, Gamma, beta, 'MAT', wMAT);
