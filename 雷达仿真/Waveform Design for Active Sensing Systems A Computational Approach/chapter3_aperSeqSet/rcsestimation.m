function rcsEst = rcsestimation(Mt, Mr, N, P, scanSet, rcs, waveform, method, NTilde)
% rcsestimation: radar cross section estimation, data collected at
% multiple positions
%
%   rcsEst = rcsestimation2(Mt, Mr, N, P, scanSet, rcs, waveform, method)
%   Mt: number of transmit antennas
%   Mr: number of receive antennas, Mr<Mt to use capon
%   N: number of subpulses transmitted by each antenna
%   P: [0 P-1] is the range of SAR imaging
%   scanSet: angle of imaging, K-by-1, in the unit of degree
%   rcs: the matrix of radar cross section, P-by-K
%   waveform: N-by-Mt, probing waveforms
%   method: 'ls', 'capon'
%   NTilde: number of positions where data are collected, each new position
%   is shifted by a distance of Mt*Mr/2 wavelength
%   rcsEst: estimated rcs, P-by-K
%
%   10/20/2008

K = length(scanSet);
adjDis = Mt*Mr/10; % the distance (unit: wavelength) between two adjacent positions where data are collected

% transmit antennas are spaced Mr*lambda/2
strVecTr = exp(-j * 2*pi * (0:(Mt-1))' * Mr/2 * sin(scanSet'*pi/180)); % Mt-by-K

% receive antennas are spaced lambda/2
strVecRe = exp(-j * 2*pi * (0:(Mr-1))' * 1/2 * sin(scanSet'*pi/180)); % Mr-by-K

X = waveform; % N-by-Mt
XTilde = [X; zeros(P-1, Mt)]; % (N+P-1)-by-Mt
noisePower = 0.001;

% received data
DH = zeros(Mr, N+P-1, NTilde);
for n = 1:NTilde
    for p = 0:(P-1)
        tmpK = zeros(Mr, Mt);
        for k = 1:K
            tmpK = tmpK + rcs(p+1, k) * strVecRe(:, k) * strVecTr(:, k).' * ...
                exp(-j * 2*pi * (n-1)*adjDis * sin(scanSet(k)*pi/180) * 2); % phase shift here
        end
        DH(:,:,n) = DH(:,:,n) + tmpK * XTilde' * sarJJ(N, P, p);
    end
    DH(:,:,n) = DH(:,:,n) + sqrt(noisePower) * (randn(Mr, N+P-1) + j * randn(Mr, N+P-1))/sqrt(2);
end

% estimate RCS
rcsEst = zeros(P, K);
for p = 0:(P-1)
    Jp = sarJJ(N, P, p);
    XTildeMF = Jp' * XTilde * inv(XTilde' * XTilde); % Mt-by-Mt
    DpTildeH = zeros(Mr, Mt, NTilde);
    for n = 1:NTilde
        DpTildeH(:,:,n) = DH(:,:,n) * XTildeMF; % Mr-by-Mt
    end
    DpTildeH = reshape(DpTildeH, [Mr Mt*NTilde]); % Mr-by-Mt*NTilde
    RpInv = inv(DpTildeH * DpTildeH') / (Mt * NTilde); % Mr-by-Mr
    for k = 1:K
        apk = strVecRe(:, k); % Mr-by-1
        phaseShift = zeros(Mt, Mt, NTilde); % steering vector phase shift for different data collection locations
        for n = 1:NTilde
            phaseShift(:,:,n) = eye(Mt) * exp(-j * 2*pi * (n-1)*adjDis * sin(scanSet(k)*pi/180) * 2); % phase shift here
        end
        phaseShift = reshape(phaseShift, [Mt Mt*NTilde]);
        bpkTilde = ((strVecTr(:,k)).' * phaseShift)'; % Mt*NTilde-by-1
        % bpkTilde = ((strVecTr(:, k)).' * XTilde' * Jp * XTildeMF * phaseShift)'; % Mt*NTilde-by-1
        if strcmp(method, 'ls')
            rcsEst(p+1, k) = (apk' * DpTildeH * bpkTilde) / (norm(apk) * norm(bpkTilde))^2;
        elseif strcmp(method, 'capon')
            rcsEst(p+1, k) = (apk' * RpInv * DpTildeH * bpkTilde) / ((norm(bpkTilde))^2 * apk' * RpInv * apk);
        else
            error('unknown method');
        end
    end
end