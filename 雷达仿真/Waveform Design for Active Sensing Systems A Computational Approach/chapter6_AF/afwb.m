function afwb(fc, uAmpPhase, fCode, sr, vMax)
% afwb(fc, uAmpPhase, fCode, sr, vMax): plot the ambiguity function of a
% pulse, under the wideband assumption
% afwb(fc, uAmpPhase), ambfunwb(fc, uAmpPhase, fCode) and 
% afwb(fc, uAmpPhase, fCode) are all OK
% The ambiguity function \chi(tau,eta) = 
%   \sqrt{eta} \int u(t) u^*(eta(t-tau)) exp(-j2\pi fc (eta-1)t) dt
% where eta = (1+v/c)(1-v/c), v is the speed of the target, c is the speed
% of the signal in certain material (e.g. 1482 m/s for sound in water)
%   fc: carrier frequency in units of 1/tb (tb is the length of one
%   subpulse)
%   uAmpPhase: N-by-1, code of the amplitude and phase
%   fCode: N-by-1, frequency code in units of 1/tb
%   sr: over-sampling rate (# of samples per subpulse)
%   vMax: maximum target speed (normalized by the speed of light or sound, denoted as c)
%
%   Some examples:
%   E.g. 1 (phase coding):
%   x = cansiso(100);
%   ambfunwb(fc, x);
%   ambfunwb(fc, x, zeros(100,1), 3*fc, 0.5);
%   E.g. 2 (LFM):
%   deltaF = 0.0031; (it should be small, because we use step frequency to
%   approximate linear frequency)
%   ambfunwb(fc, ones(51,1), deltaF * (-25:25)', 10, 0.2);
%   E.g. 3 (LFM):
%   ambfunwb(10, ones(20,1), (linspace(-1,1,20))', 40, 0.05);
%   ambfunwb(4, ones(20,1), (linspace(-1,1,20))', 20, 0.125);
%   E.g. 4 (Zhaofu's SST paper)
%   x = cansiso(400);
%   fc = 900/B = 4.5;
%   ambfun(4.5, x, zeros(400,1), 10, 0.0028);

N = length(uAmpPhase); % original length of the code

if nargin <= 3 % no sr specified
    vMax = 0.05;
    sr = 2 * fc; % at least 2fc   
else
    sr = ceil(sr);
    if nargin == 4
        vMax = 0.1;
    end
end

K = 100; % # of positive speed grid points for calculation  
dv = vMax / K; % Speed grid step (in unit of 1/c)
dt = 1 / sr; % Delay time grid step (in units of tb)

% generate the over-sampled signal uSamp (each element repeated sr times)
NSamp = N * sr;
uSamp = (reshape(ones(sr,N) * diag(uAmpPhase), [1 NSamp])).'; % NSamp-by-1

% calculate the amplitude, phase and frequency information
uAmp = abs(uSamp); % NSamp-by-1, amplitude
uPhase = angle(uSamp); % phase
if nargin >= 3 % frequency coding
    uPhase = uPhase + 2*pi * dt * (cumsum(reshape(ones(sr,N) * diag(fCode),...
        [1 NSamp]))).';
    uSamp = uAmp .* exp(1i * uPhase);
end

t = (0:1:NSamp-1)' * dt; % times corresponding to samples
v = (0:1:K-1)' * dv; % speed

% let's calculate half of the ambiguity function for v > 0
% u is the orignal signal
% v = u(eta t)

AmbFun = zeros(2*NSamp-1, K);
for k = 1:K
    disp(['k = ' num2str(k)]);
    et = (1 + v(k)) / (1 - v(k));
    tCrs = t / et; % NSamp-by-1 (the received signal is compressed)
    % u(eta t) is u(t) with the sampling time changed from dt to dt/eta
    tCompare = repmat(t', [NSamp 1]) - repmat(tCrs, [1 NSamp]);
    [tmp indexTimeClose] = min(abs(tCompare));
    % the compressed signal is represented by uSamp under the assumption
    % that the sampling interval is dt/eta. vSamp is the samples of this
    % underlying compressed signal at regular t = [0 1 ... NSamp-1] dt
    vSamp = uSamp(indexTimeClose);
    indexTimeOutRange = find(indexTimeClose == NSamp);
    if length(indexTimeOutRange) >= 2 % samples outside the signal duration
        vSamp(indexTimeOutRange(2:end)) = 0;
    end    
    AmbFun(:,k) = sqrt(et) * xcorr(uSamp .* exp(-1i*2*pi*fc*(et-1)*t), ...
        vSamp); % 2NSamp-1 by 1
end
AmbFun = abs(AmbFun);
AmbFun = AmbFun.' / max(AmbFun(:)); % K by 2NSamp-1

% plot the ambiguity function
tDelay = [-flipud(t); t(2:end)]; % 2NSamp-1 by 1, delay times
tDelayPlot = [-N; tDelay; N]; % add two end points
% eta = (1 + v) ./ (1 - v); % K-by-1
% etaPlot = [1; eta]; % for better visual effects
vPlot = [0; v];
AmbFunPlot = [zeros(1,2*NSamp-1); AmbFun]; % AmbFun is K-by-(2NSamp-1)
AmbFunPlot = [zeros(K+1,1) AmbFunPlot zeros(K+1,1)]; % K+1 by 2NSamp+1
figure;
mesh(tDelayPlot/N, vPlot, AmbFunPlot); hold on;
surface(tDelayPlot/N, [0 0], AmbFunPlot(1:2,:)); hold off;
colormap('default');
axis([-1 1 0 vMax 0 1]);
xlabel('\tau / T');
ylabel('v / c');
%title('$|\bar{\chi}_B(\tau,v)|$ (dB)', 'Interpreter', 'LaTex');
myboldify;

% % plot the auto-correlation function
% figure;
% plot(tDelayPlot/N, 20*log10(AmbFunPlot(2,:)));
% axis([-1 1 -60 0]);
% xlabel('\tau / T');
% ylabel('|r(\tau)|');
% title('Auto-correlation (dB)');
% myboldify;

% % plot the 0-delay function
% figure;
% plot(vPlot(2:end), 20*log10(AmbFunPlot(2:end, NSamp+1)));
% axis([0 vMax -inf 0]);
% xlabel('v / c');
% title('|\chi(0,v)|');
% myboldify;

% plot the log-scale imagesc ambiguity function (all four quadrants)
figure;
map = flipud(colormap('hot')); % inverse hot map
AmbFunPart2 = fliplr(flipud(AmbFun(2:end,:))); % AF when v<0 (eta<1)
% AF(tau, eta) = AF*(-eta tau, 1/eta) when eta<1
% AF(tau, v) = AF*(-eta tau, -v) when v<0
vMinus = (-K+1:-1)' * dv;
for k = 1:(K-1)
    vNow = vMinus(k);
    et = (1 + vNow) / (1 - vNow); % <1
    AmbFunPart2(k,:) = AmbFunPart2(k, NSamp + round(et * (-NSamp+1:NSamp-1)));
    % from AF(-tau, -v) to AF(-eta tau, -v)
end
AmbFunFull = [AmbFunPart2; AmbFun]; % 2K-1 by 2NSamp-1
vFull = (-K+1:1:K-1)' * dv;
%etaFull = (1 + vFull) ./ (1 - vFull);
imagesc(tDelay/N, vFull, 20*log10(AmbFunFull), [-40 0]);
set(gca, 'YDir', 'normal');
axis([-inf inf -vMax vMax]);
colormap(map);
xlabel('\tau / T');
ylabel('v / c');
%title('$|\bar{\chi}_B(\tau,v)|$ (dB)', 'Interpreter', 'LaTex');
colorbar;
myboldify;

% show the peak sidelobe
peakSidelobe = afsidelobe(AmbFun);
disp(['Peak sidelobe: ' num2str(peakSidelobe)]);