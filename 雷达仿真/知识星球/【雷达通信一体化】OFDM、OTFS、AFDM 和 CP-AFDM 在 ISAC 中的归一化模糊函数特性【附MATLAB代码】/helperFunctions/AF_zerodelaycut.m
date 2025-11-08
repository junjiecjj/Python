function [nu_os, AdB_os, A_os] = AF_zerodelaycut(s, OS, doppler_span)
    % Zero-delay cut A(0,nu) evaluated densely and exactly
    % INPUTS
    %   s            : length-N complex/real vector (row or column)
    %   OS           : Doppler oversampling factor (e.g., 32, 64, 128)
    %   doppler_span : plot span in normalized units; evaluate nu in [-span, +span]
    % OUTPUTS
    %   nu_dense  : 1×Nd vector of normalized Doppler points (includes 0 exactly)
    %   AdB_dense : 1×Nd zero-delay cut in dB, normalized (peak = 0 dB)
    %
    % =========================================================================
    % Author    : Dr. (Eric) Hyeon Seok Rou
    % Version   : v1.0
    % Date      : Oct 13, 2025
    % =========================================================================
    
    if nargin < 2 || isempty(OS), OS = 64; end
    if nargin < 3 || isempty(doppler_span), doppler_span = 5; end
    if OS < 1, OS = 1; end
    
    s = s(:);
    N = length(s);
    
    % Energy of s: Es = sum |s[n]|^2
    Es = 0.0;
    for n = 1:N
        sr = real(s(n)); si = imag(s(n));
        Es = Es + (sr*sr + si*si);
    end
    if Es == 0
        nu_os = linspace(-doppler_span, doppler_span, max(2, OS*N));
        AdB_os = dB_floor*ones(size(nu_os));
        return;
    end
    
    % Number of dense Doppler samples
    Nd = max(2, OS * N);
    
    % Build symmetric Doppler grid that GUARANTEES nu = 0 is on-grid
    if mod(Nd,2)==1
        nu_os = linspace(-doppler_span, +doppler_span, Nd);
    else
        Nd_eff   = Nd + 1;
        nu_full  = linspace(-doppler_span, +doppler_span, Nd_eff);
        nu_os = nu_full(1:Nd); % drop last endpoint (+span) so 0 is included
    end
    
    % A(0,nu) = sum_{n=0}^{N-1} |s[n]|^2 * exp(-j 2 pi nu n)
    A0 = complex(zeros(1, numel(nu_os)));
    for k = 1:numel(nu_os)
        nu_k = nu_os(k);
        accR = 0.0; accI = 0.0;
        for n = 1:N
            % |s[n]|^2
            sr = real(s(n)); si = imag(s(n));
            p  = sr*sr + si*si;
    
            % exp(-j 2 pi nu_k (n-1))  [use 0-based time index]
            phase = -2*pi*nu_k*(n-1);
            c = cos(phase); d = sin(phase);
    
            % accumulate p * (c + j d)
            accR = accR + p*c;
            accI = accI + p*d;
        end
        A0(k) = complex(accR, accI);
    end
    A_os = A0;
    A0abs = abs(A0);
    A0abs = A0abs ./ (max(A0abs) + eps);
    AdB_os = 20*log10(A0abs + eps);
end