function [s w mse msebound] = crew(N, algname, jammertype, PAR, s0)
% crew: cognitive receiver and waveform
% [s w mse msebound] = crew(N, algname, jammertype, PAR, s0)
%
%   N: length of the probing sequence or receive filter
%   algname: 'CAN', 'IVC', 'MAT', 'GRA', 'CYC', 'FRE' that represent
%      respectively CAN, CAN-IV, CREW(mat), CREW(gra), CREW(cyc), CREW(fre)
%   jammertype: 'white', 'spot', 'barrage'
%   PAR: (optional) any number in [1,N]. Not meaningful for 'GRA' which
%      only deals with the case of PAR=1
%   s0: (optional) N-by-1, the initialization sequence
%
%   s: N-by-1, the obtained probing sequence
%   w: N-by-1, the obtained receive filter
%   mse: mean-squared error of the RCS estimate
%   msebound: the lower bound for mse. Only meaningful for 'MAT' and 'FRE'

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

% run the algorithm
msebound = 0; % only appies to MAT and FRE
switch algname
    case 'CAN'
        s = cansisopar(PAR, N, s0);
        w = s;
        R = Rcalc(s, Gamma, beta);
        mse = real(w' * R * w) / (abs(w'*s))^2;
    case 'MAT'
        [s w mse msebound] = cyc_freq_mat(N, Gamma, beta, PAR, s0);        
    case 'IVC'
        s = cansisopar(PAR, N, s0);
        R = Rcalc(s, Gamma, beta);
        w = R \ s;
        mse = 1 / real(s' * w); % 1/(s' * inv(R) * s)
    case 'CYC'
        [s w mse] = cyc_direct(N, Gamma, beta, PAR, s0);
    case 'GRA'
        [s w mse] = grad_direct(N, Gamma, beta, s0); % PAR=1
    case 'FRE'
        [s w mse msebound] = cyc_freq(N, Gamma, beta, PAR, s0);
%     case 'FRW' % FRE with weights -- a failed approach
%         [s w mse msebound] = cyc_freq_weight(N, Gamma, beta, PAR, s0);
%     case 'FRZ' % FRE with zero-padding constraint
%         [s w mse] = cyc_freq_sdp(N, Gamma, beta, PAR, s0);
    otherwise
        error('unknown algorithm name');
end

%show_criterion(s, Gamma, beta, algname, w);




