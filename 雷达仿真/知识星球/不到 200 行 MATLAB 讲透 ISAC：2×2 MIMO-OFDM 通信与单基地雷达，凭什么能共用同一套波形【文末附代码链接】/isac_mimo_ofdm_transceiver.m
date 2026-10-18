%% ISAC MIMO-OFDM transceiver: communications + monostatic sensing
% Self-contained base-MATLAB simulation (no toolbox functions required).
% It models a 2x2 spatial-multiplexing communication link and a 2x2
% time-division MIMO radar burst sharing the same OFDM numerology.

clear; clc; close all;
rng(17, 'twister');

%% System configuration
c = 299792458;
fc = 5.9e9;                         % Carrier frequency [Hz]
lambda = c/fc;
Nt = 2; Nr = 2;                     % Tx/Rx antennas
Nfft = 64;
subcarrierSpacing = 15e3;
Fs = Nfft*subcarrierSpacing;
Ncp = 16;                           % CP: all modeled delays are inside it
Tsym = (Nfft + Ncp)/Fs;
snrDb = 22;

%% Communications frame and transmitted QPSK streams
NcommSym = 20;
pilotSym = 1:Nt;                    % One full-band, orthogonal pilot symbol/Tx
dataSym = (Nt+1):NcommSym;
NdataBits = 2*Nfft*numel(dataSym)*Nt;
txBits = randi([0 1], NdataBits, 1);
txData = qpskMod(txBits);

Xcomm = zeros(Nfft, NcommSym, Nt);
for tx = 1:Nt
    % A separate all-subcarrier pilot symbol makes the MIMO channel observable.
    Xcomm(:, pilotSym(tx), tx) = exp(1j*2*pi*(0:Nfft-1)'*(tx+1)/Nfft);
end
Xcomm(:, dataSym, :) = reshape(txData, Nfft, numel(dataSym), Nt);
txCommWave = ofdmModulate(Xcomm, Nfft, Ncp);

%% Frequency-selective MIMO communication channel
% The first path is at zero delay, so FFT-window timing is exact here.
tapDelays = [0 3 7];
tapPowerDb = [0 -5 -10];
hComm = makeMimoChannel(Nr, Nt, tapDelays, tapPowerDb);
rxCommWaveClean = applyMimoChannel(txCommWave, hComm);
signalPower = mean(abs(rxCommWaveClean(:)).^2);
noiseVariance = signalPower/10^(snrDb/10);
rxCommWave = rxCommWaveClean + sqrt(noiseVariance/2) * ...
    (randn(size(rxCommWaveClean)) + 1j*randn(size(rxCommWaveClean)));

%% Communication receiver: CP removal, FFT, pilot LS estimate, MMSE MIMO equalizer
Ycomm = ofdmDemodulate(rxCommWave, Nfft, Ncp, NcommSym);
Hest = zeros(Nfft, Nr, Nt);
for tx = 1:Nt
    for rx = 1:Nr
        Hest(:, rx, tx) = Ycomm(:, pilotSym(tx), rx) ./ Xcomm(:, pilotSym(tx), tx);
    end
end

rxData = zeros(Nfft, numel(dataSym), Nt);
for m = 1:numel(dataSym)
    symbolIndex = dataSym(m);
    for k = 1:Nfft
        Hk = squeeze(Hest(k, :, :));             % Nr-by-Nt
        yk = squeeze(Ycomm(k, symbolIndex, :));  % Nr-by-1
        Wmmse = (Hk' * Hk + noiseVariance*eye(Nt)) \ Hk';
        rxData(k, m, :) = Wmmse * yk;
    end
end
rxBits = qpskDemod(rxData(:));
ber = mean(rxBits ~= txBits);
evmRms = sqrt(mean(abs(rxData(:) - txData).^2) / mean(abs(txData).^2));
fprintf('Communication BER = %.4g, RMS EVM = %.2f %% at %.1f dB SNR\n', ...
    ber, 100*evmRms, snrDb);

%% ISAC sensing burst: TDM-MIMO OFDM radar waveform
% Only one transmitter is active per OFDM symbol. This gives every Tx an
% independent, known waveform and produces a virtual Nt*Nr sensing array.
NpulsesPerTx = 16;
NradSym = Nt*NpulsesPerTx;
Xrad = zeros(Nfft, NradSym, Nt);
for m = 1:NradSym
    activeTx = mod(m-1, Nt) + 1;
    Xrad(:, m, activeTx) = qpskMod(randi([0 1], 2*Nfft, 1));
end
txRadarWave = ofdmModulate(Xrad, Nfft, Ncp);

% Targets are deliberately quantized to sample delays for a clean, directly
% interpretable range-bin check. Their delay is less than the 16-sample CP.
targetDelaySamples = [4 10];
targetRange = targetDelaySamples*c/(2*Fs);
targetVelocity = [10 -6];          % Radial velocity [m/s]
targetDoppler = 2*targetVelocity*fc/c;
targetAzimuthDeg = [18 -25];
targetAmplitude = [0.85 0.45] .* exp(1j*[0.4 -1.1]);
rxRadarWaveClean = targetEcho(txRadarWave, targetDelaySamples, targetDoppler, ...
    targetAzimuthDeg, targetAmplitude, fc, Fs, Nt, Nr);
radarSignalPower = mean(abs(rxRadarWaveClean(:)).^2);
radarNoiseVariance = radarSignalPower/10^(snrDb/10);
rxRadarWave = rxRadarWaveClean + sqrt(radarNoiseVariance/2) * ...
    (randn(size(rxRadarWaveClean)) + 1j*randn(size(rxRadarWaveClean)));

%% Radar receiver: OFDM channel division, range IFFT, Doppler FFT and combining
Yrad = ofdmDemodulate(rxRadarWave, Nfft, Ncp, NradSym);
rangeCube = zeros(Nfft, NpulsesPerTx, Nr, Nt);
for tx = 1:Nt
    symbolsForTx = tx:Nt:NradSym;
    for rx = 1:Nr
        targetResponse = Yrad(:, symbolsForTx, rx) ./ Xrad(:, symbolsForTx, tx);
        rangeCube(:, :, rx, tx) = ifft(targetResponse, Nfft, 1);
    end
end
rdCube = fftshift(fft(rangeCube, NpulsesPerTx, 2), 2);
rdPower = squeeze(sum(sum(abs(rdCube).^2, 4), 3));
rdPowerDb = 10*log10(rdPower/max(rdPower(:)) + eps);

rangeAxis = (0:Nfft-1)' * c/(2*Fs);
PRI = Nt*Tsym;                      % Repetition interval for each TDM Tx
dopplerAxis = ((-NpulsesPerTx/2):(NpulsesPerTx/2-1))/(NpulsesPerTx*PRI);
velocityAxis = dopplerAxis*c/(2*fc);
[~, peakIndex] = max(rdPower(:));
[rangeBin, velocityBin] = ind2sub(size(rdPower), peakIndex);
fprintf('Strongest sensing peak: range %.1f m, velocity %.2f m/s\n', ...
    rangeAxis(rangeBin), velocityAxis(velocityBin));
fprintf('Injected targets: [%.1f, %.1f] m and [%.1f, %.1f] m/s\n', ...
    targetRange(1), targetRange(2), targetVelocity(1), targetVelocity(2));

%% Validation plots
figure('Color','w', 'Name','ISAC MIMO-OFDM results', 'Position',[100 100 1150 720]);
subplot(2,2,1);
plot(real(txData(1:500)), imag(txData(1:500)), 'o', 'MarkerSize', 4); grid on; axis square;
xlabel('In-phase'); ylabel('Quadrature'); title('Transmit QPSK constellation'); xlim([-1.5 1.5]); ylim([-1.5 1.5]);
subplot(2,2,2);
plot(real(rxData(:)), imag(rxData(:)), '.', 'MarkerSize', 7); grid on; axis square;
xlabel('In-phase'); ylabel('Quadrature'); title(sprintf('Equalized data: BER %.3g, EVM %.2f%%', ber, 100*evmRms)); xlim([-1.5 1.5]); ylim([-1.5 1.5]);
subplot(2,2,3);
imagesc(velocityAxis, rangeAxis, rdPowerDb); axis xy; colorbar; caxis([-35 0]);
xlabel('Radial velocity (m/s)'); ylabel('Range (m)'); title('Combined MIMO range-Doppler map (dB)');
hold on; plot(targetVelocity, targetRange, 'wo', 'LineWidth', 1.5, 'MarkerSize', 8); hold off;
subplot(2,2,4);
plot(rangeAxis, max(rdPowerDb, [], 2), 'LineWidth', 1.2); grid on; hold on;
xline(targetRange(1), '--'); xline(targetRange(2), '--'); hold off;
xlabel('Range (m)'); ylabel('Peak Doppler power (dB)'); title('Range profile'); ylim([-45 2]);

outputPng = fullfile(fileparts(mfilename('fullpath')), 'isac_mimo_ofdm_results.png');
exportgraphics(gcf, outputPng, 'Resolution', 160);
fprintf('Saved result plot to: %s\n', outputPng);

%% Local functions
function symbols = qpskMod(bits)
bits = bits(:);
assert(mod(numel(bits),2) == 0, 'QPSK requires an even number of bits.');
symbols = ((1 - 2*bits(1:2:end)) + 1j*(1 - 2*bits(2:2:end))) / sqrt(2);
end

function bits = qpskDemod(symbols)
symbols = symbols(:);
bits = zeros(2*numel(symbols), 1);
bits(1:2:end) = real(symbols) < 0;
bits(2:2:end) = imag(symbols) < 0;
end

function wave = ofdmModulate(grid, Nfft, Ncp)
[~, Nsym, Nant] = size(grid);
timeNoCp = ifft(grid, Nfft, 1) * sqrt(Nfft);    % Unit-power FFT convention
timeWithCp = [timeNoCp(end-Ncp+1:end,:,:); timeNoCp];
wave = reshape(timeWithCp, (Nfft+Ncp)*Nsym, Nant);
end

function grid = ofdmDemodulate(wave, Nfft, Ncp, Nsym)
Nant = size(wave,2);
assert(size(wave,1) == (Nfft+Ncp)*Nsym, 'Unexpected received frame length.');
blocks = reshape(wave, Nfft+Ncp, Nsym, Nant);
grid = fft(blocks(Ncp+1:end,:,:), Nfft, 1) / sqrt(Nfft);
end

function h = makeMimoChannel(Nr, Nt, delays, powerDb)
L = max(delays) + 1;
h = zeros(Nr, Nt, L);
for p = 1:numel(delays)
    spatialTap = (randn(Nr,Nt) + 1j*randn(Nr,Nt)) / sqrt(2);
    h(:,:,delays(p)+1) = sqrt(10^(powerDb(p)/10)) * spatialTap;
end
end

function y = applyMimoChannel(x, h)
[Nr, Nt, ~] = size(h);
Ns = size(x,1);
y = zeros(Ns, Nr);
for rx = 1:Nr
    for tx = 1:Nt
        response = squeeze(h(rx,tx,:));
        filtered = conv(x(:,tx), response);
        y(:,rx) = y(:,rx) + filtered(1:Ns);
    end
end
end

function y = targetEcho(x, delays, dopplers, azimuthDeg, amplitudes, fc, Fs, Nt, Nr)
c = 299792458;
lambda = c/fc;
elementSpacing = lambda/2;
Ns = size(x,1);
n = (0:Ns-1)';
y = zeros(Ns, Nr);
for q = 1:numel(delays)
    theta = deg2rad(azimuthDeg(q));
    aTx = exp(1j*2*pi*elementSpacing/lambda*(0:Nt-1)'*sin(theta));
    aRx = exp(1j*2*pi*elementSpacing/lambda*(0:Nr-1)'*sin(theta));
    delayed = zeros(size(x));
    delayed(delays(q)+1:end,:) = x(1:end-delays(q),:);
    transmittedAtTarget = delayed * aTx;
    dopplerPhase = exp(1j*2*pi*dopplers(q)*n/Fs);
    for rx = 1:Nr
        y(:,rx) = y(:,rx) + amplitudes(q)*aRx(rx)*dopplerPhase .* transmittedAtTarget;
    end
end
end
