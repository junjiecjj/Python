function [mseTrue mseFalse mseRobust] = ...
    crew_robust(N, algname, jammertype, PAR, s0)
% crew: cognitive receiver and waveform, robust design
% [mseTrue mseFalse mseRobust] = crew(N, algname, jammertype, PAR, s0)
%
%   N: length of the probing sequence or receive filter
%   algname: 'CAN', 'IVC', 'MAT', 'GRA', 'CYC', 'FRE' that represent
%      respectively CAN, CAN-IV, CREW(mat), CREW(gra), CREW(cyc), CREW(fre)
%      However, currently only 'FRE' is taken care of
%   jammertype: 'white', 'spot', 'barrage'
%   PAR: (optional) any number in [1,N]. Not meaningful for 'GRA' which
%      only deals with the case of PAR=1
%   s0: (optional) N-by-1, the initialization sequence
%
%   mseTrue: MSE when the true Gamma is known
%   mseFalse: MSE when the info on Gamma is imprecise
%   mseRobust: MSE when the info on Gamma is imprecise but with robust
%   design

% at least three input arguments
if nargin < 3
    error('crew: not enough input arguments');
elseif nargin < 5
    s0 = exp(1i * 2*pi * rand(N,1));
    if nargin < 4
        PAR = 1;
    end
end

% the following settings are hard-coded here and must be updated manually
sigma2 = 0.1; % noise power
sigma2jammer = 100; % jammer power
beta = 1; % expected value of |alpha|^2 where alpha is the RCS

% three jammer types
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
        
        f1 = 0.25;
        Num = 2*N-1;
        power = zeros(Num,1);
        power(floor(f1*Num)+1) = 1;
        r = ifft(power); % [r(0) r(1) ... r(N-1) r*(N-1) ... r*(1)].'
        r = r/abs(r(1)) * sigma2jammer;
        Gamma_false = toeplitz(r(1:N), [r(1) r(2*N-1:-1:N+1).']) ...
            + sigma2 * eye(N);
        
        f1 = 0.15; f2 = 0.25;
        Num = 2*N-1;
        power = zeros(Num,1);
        power(floor(f1*Num)+1 : floor(f2*Num)) = 1;
        r = ifft(power); % [r(0) r(1) ... r(N-1) r*(N-1) ... r*(1)].'
        r = r/abs(r(1)) * sigma2jammer;
        Gamma_robust = toeplitz(r(1:N), [r(1) r(2*N-1:-1:N+1).']) ...
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
        
        f1 = 0.25; f2 = 0.35;
        Num = 2*N-1;
        power = zeros(Num,1);
        power(floor(f1*Num)+1 : floor(f2*Num)) = 1;
        r = ifft(power); % [r(0) r(1) ... r(N-1) r*(N-1) ... r*(1)].'
        r = r/abs(r(1)) * sigma2jammer;
        Gamma_false = toeplitz(r(1:N), [r(1) r(2*N-1:-1:N+1).']) ...
            + sigma2 * eye(N);
        
        f1 = 0.15; f2 = 0.35;
        Num = 2*N-1;
        power = zeros(Num,1);
        power(floor(f1*Num)+1 : floor(f2*Num)) = 1;
        r = ifft(power); % [r(0) r(1) ... r(N-1) r*(N-1) ... r*(1)].'
        r = r/abs(r(1)) * sigma2jammer;
        Gamma_robust = toeplitz(r(1:N), [r(1) r(2*N-1:-1:N+1).']) ...
            + sigma2 * eye(N);
    otherwise
        error('Unrecognized jammer type');
end

% run the algorithm
switch algname
    case 'FRE'
        [s w mseTrue] = cyc_freq(N, Gamma, beta, PAR, s0);
        [s w] = cyc_freq(N, Gamma_false, beta, PAR, s0);
        R = Rcalc(s, Gamma, beta);
        mseFalse = real(w' * R * w) / (abs(w'*s))^2;
        [s w] = cyc_freq(N, Gamma_robust, beta, PAR, s0);
        R = Rcalc(s, Gamma, beta);
        mseRobust = real(w' * R * w) / (abs(w'*s))^2;
    otherwise
        error('unknown algorithm name');
end





