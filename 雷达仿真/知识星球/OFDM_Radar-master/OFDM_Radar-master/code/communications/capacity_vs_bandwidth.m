%% Investigating the relationship between capacity, bandwidth, and SNR:
%
%
% Author: Dylan Boland (Student)

IEEEstandard = "802.11a"; % the IEEE standard that we want to simulate...

% Let us define the OFDM system parameters based on the IEEE 802 standard
% in use:
switch (IEEEstandard)
    case "802.11a"
        To = 4e-6; % the OFDM symbol duration (period)
        fC = 5.5e9; % 802.11a uses the 5 GHz band
        N = 52; % 48 for data, and 4 pilot tones
        numPilotTones = 4; % no. pilot tones used
        M = 68; % no. of OFDM symbols in a frame or burst
        G = 1/4; % the guard interval fraction for 802.11a 
        subcarrierSpacing = 312.5e3; % subcarrier spacing for 802.11a
    case "802.11g"
        To = 4e-6; % the OFDM symbol duration (period)
        fC = 2.4e9; % 802.11g uses the 2.4 GHz band
        N = 52; % 48 for data, and 4 pilot tones
        numPilotTones = 4; % no. pilot tones used
        M = 7; % no. of OFDM symbols in a frame or burst
        G = 1/4; % the guard interval fraction for 802.11g
        subcarrierSpacing = 312.5e3; % subcarrier spacing for 802.11g
end

% Encoder:
r = 1/2; % the code rate used at the Tx
txTrellis = poly2trellis(7, [171 133]);
traceBackLength = 34;

% Generate bit stream:
modOrder = 16; % order of the modulation scheme being used
bitsPerSymbol = log2(modOrder); % the number of bits each symbol will represent...
numBits = (N/2 - 1)*(M)*(bitsPerSymbol)*r; % number of bits we need to generate
txBits = randi([0 1], numBits, 1); % our transmit bit stream

% Encode the data:
encodedData = convenc(txBits, txTrellis);

% Modulate the bits: map the bits to symbols:
qamSymbols = qammod(encodedData, modOrder, "InputType", "bit", "UnitAveragePower", true);

% The transmit frame (Ftx). Row 1 is left idle, as this is the DC
% subcarrier:
Ftx = zeros(N, M);
Ftx(2:N/2, :) = reshape(qamSymbols, (N/2-1), M);

% Impose Hermitian Symmetry:
Ftx(N/2+2:end, :) = conj(flip(Ftx(2:N/2, :), 1));

% Compute the FFT of each column of Ftx:
s = ifft(Ftx);

% Add the cyclic prefix to combat intersymbol interference (ISI):
cpLen = G*N; % the number of samples in the cyclic prefix
s = [s(end-cpLen+1:end, :); s];

% Form the transmit sequence by concatenating the OFDM symbols:
s = s(:); % the transmit sequence (s)

bandwidth = N*subcarrierSpacing; % bandwidth of system in units of Hz
SNRdB = 1:1:30; % SNR values in dB
SNR = 10.^(SNRdB/10); % SNR values in linear units

r = zeros(length(SNR), length(s)); % a matrix to hold receive sequences...

BER = zeros(length(SNR), 1); % a vector to store BER values...

% Let us form the receive sequence:
for i = 1:1:length(SNR)
    r(i, :) = awgn(s, SNR(i), 'measured');
    Frx = reshape(r(i, :), N+cpLen, M);
    % remove the CP from each symbol:
    Frx = Frx(cpLen+1:end, :);
    % take the FFT of each column:
    Frx = fft(Frx);
    % now convert complex symbols to bits
    Frx = Frx(2:N/2, :); % get the payload
    Frx = Frx(:);
    encodedDataRx = qamdemod(Frx, modOrder, "OutputType", "bit", "UnitAveragePower", true);
    rxBits = vitdec(encodedDataRx, txTrellis, traceBackLength, 'trunc', 'hard');
    numBitErrors = numBits - sum(rxBits == txBits);
    BER(i) = numBitErrors/numBits;
end