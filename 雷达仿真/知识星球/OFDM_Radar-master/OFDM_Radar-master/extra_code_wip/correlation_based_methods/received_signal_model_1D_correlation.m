%% Program to model r(t), the received signal, after sending out s(t):
N = 64; % The number of subcarriers being used
frameSize = 256; % The frame size
M = 16; % Order of QAM scheme being used
d = (0:M-1)';

% General constants:
c = 3e8; % The speed of light

% OFDM system parameters:
To = 4e-6; % The OFDM symbol period
fC = 5.5e9; % The frequency of the centre subcarrier
subcarrierSpacing = 312.5e3; % The subcarrier spacing

% Parameters of targets:
targetDistances = [40]; % Only 1 target for now
targetRelVelocities = [105];
targetRCSs = [10];
bVec = sqrt((c.*targetRCSs)./((4*pi)^3*(targetDistances.^4)*(fC^2))); % The attenuations associated with the various targets...

numTargets = length(targetDistances);

% Doppler shift and time delay values:
dopplerShifts = 2*fC*(targetRelVelocities)/c;
timeDelays = 2*(targetDistances)/c;

modulationType = "QPSK";

switch(modulationType)
    case "QAM"
        signalSet = qammod(d, M); % Generate the signal set for the relevant QAM scheme (16, 32, 64 etc.)     
    case "QPSK"
        signalSet = pskmod(d, M);
    otherwise
        disp("Check the status of 'modulationType' above.")
        return % Exit the program early
end

Ftx = zeros(N/2, frameSize); % Half of the empty transmit frame

for row = 2:N/2 % We start at row 2, as row 1 is empty for the DC subcarrier...
    Ftx(row, :) = randsample(signalSet, frameSize, true);
end

% The full transmit frame. Here we are imposing Hermitian symmetry, so that
% the output from the IFFT block is real-valued...
Ftx = [Ftx; zeros(1, frameSize); conj(flip(Ftx(2:end, :), 1))];

% Creating the OFDM transmit frame:
cpLen = 10;
ofdmFrame = ifft(Ftx); % Passing each column of Ftx into the IFFT block
ofdmFrame = [ofdmFrame(end-cpLen+1:end, :); ofdmFrame]; % Adding the Cyclic prefix (CP) to each column (symbol)...

% Concatenating the symbols, as is done before transmission:
sTx = ofdmFrame(:); % I think this signal would be sampled by a DAC at Tx side. The DAC would then
% output an analog signal which would be converted to an EM wave at the Tx
% antenna...
fs = 300e6; % The sampling rate of our imagined ADC... thanks Sammy for the advice on this!
% Padding length calculation: a good portion of the beginning of our
% received signal, say, r, will be 0s: this is because we have to wait some
% time before the reflection off the target arrive at the OFDM beacon...
padLength = round(timeDelays/(1/fs));
n = 0:padLength + length(sTx)-1; % discrete time vector...
t = n*(1/fs); % The real-time values...

r = zeros(1, padLength + length(sTx)); % The received signal...

z = wgn(1, padLength + length(sTx), -20);
% Next, we compute the values of the received signal vector, r: since there
% is a delay between when we transmit s(t) and receive r(t), the first few
% samples of r(t) at the receiver will be 0s... as the reflection off the target is
% yet to reach us. The amount of 0s will be equal to the total time delay
% divided by the sampling period (which is the same as the time delay
% multiplied by the sampling frequency... ):
for j = padLength+1:length(r) % We start at index "padLength+1" for the reason stated above...
    r(j) = sTx(j-padLength)*cos(2*pi*dopplerShifts*t(j));
end

r = r + z; % Updating the received signal, so that is has noise added on top of it...
[c, lags] = xcorr(sTx, r);
[c_max, delay_samples] = max(abs(c(:)));
c_norm = c./c_max;
c_norm_max = max(c_norm(:));
% At the receiver side we want to form the received frame from the signal
% r(t), or r(n) (which is what we get after we sample the continuous wave
% r(t) at the receiver)
ofdmFrameRx = reshape(r(length(r)-delay_samples+1:end), size(ofdmFrame, 1), size(ofdmFrame, 2));
[c2, lags2] = xcorr(ofdmFrameRx(cpLen+1:end, 2), ofdmFrame(cpLen+1:end, 2));
stem(lags2, c2)
title("\fontname{Georgia}Real-valued Cross Correlation between F_{Rx} and F_{Tx} - Column 1")
xlabel("\fontname{Georgia}Lag Values")
ylabel("\fontname{Georgia}\bfCorrelation Value")