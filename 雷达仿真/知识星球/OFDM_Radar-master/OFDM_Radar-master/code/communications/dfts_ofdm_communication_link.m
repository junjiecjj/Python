%% OFDM communication link (for different IEEE 802.11a standards)

% Let us define some constants that decide the colours used in the graphs:
txSymbolsColour = [0.6 0.8 0];
rxSymbolsColour1 = [0.6 0.2 1];
rxSymbolsColour2 = [0.9 0.1 0.3];

% Boolean variable to control whether a channel with a long memory or
% maximum delay is used:
useLongDelayChannel = false;

% Boolean variable to control whether cyclic prefix is used or not:
useCP = true;

IEEEstandard = "802.11a"; % the IEEE standard that we want to simulate...

% Let us define the OFDM system parameters based on the IEEE 802 standard
% in use:
switch (IEEEstandard)
    case "802.11a"
        subcarrierSpacing = 312.5e3; % subcarrier spacing for 802.11a
        Tu = 1/subcarrierSpacing; % the "useful" symbol period
        G = 1/4; % the guard interval fraction
        Tg = G*Tu; % the guard interval time
        To = Tu + Tg; % the OFDM symbol duration (period)
        fC = 5.5e9; % 802.11a uses the 5 GHz band
        N = 52; % 48 for data, and 4 pilot tones
        numPilotTones = 4; % no. pilot tones used 
    case "802.11g"
        subcarrierSpacing = 312.5e3; % subcarrier spacing for 802.11g
        Tu = 1/subcarrierSpacing; % the "useful" symbol period
        G = 1/4; % the guard interval fraction
        Tg = G*Tu; % the guard interval time
        To = Tu + Tg; % the OFDM symbol duration (period)
        fC = 2.4e9; % 802.11a uses the 2.4 GHz band
        N = 52; % 48 for data, and 4 pilot tones
        numPilotTones = 4; % no. pilot tones used        
end

K = 6; % the scale factor between the Tx-side FFT and IFFT block
L = K*N; % size of the IFFT block used on the transmitter side

% No. OFDM symbols in frame or "burst"
M = 14;

% Mapping type:
mapType = "interleave";

% Info. on modulation scheme being used:
modType = "QAM";
modOrder = 64;
bitsPerSymbol = log2(modOrder); % no. bits per complex symbol

% Channel information:
h = [0.8; 0.51; 0.4]; % channel coefficients...
if (useLongDelayChannel)
    % by making the channel long, we can show what happens when the maximum
    % delay of the channel is greater than the cylic prefix:
    h = [h; 0.1*rand(30, 1)];
end

if (useCP == true)
    cpLen = G*L;
else
    cpLen = 0;
end
    
% Generate bit stream. We are no modelling the encoding stage  in this
% program:
numBits = (N)*bitsPerSymbol*M;
txBits = randi([0 1], numBits, 1); % the bit stream

% Modulating stage (map bits to complex symbols):
switch(modType)
    case "QAM"
        modulatedData = qammod(txBits, modOrder, "InputType", "bit", "UnitAveragePower", true);     
    otherwise
        disp("Check the status of 'modType' above.")
        return % exit the program early
end

Ftx = reshape(modulatedData, N, M); % create the transmit frame
FtxCopy = Ftx; % make a copy for Rx-side comparison

% DFT block stage:
Ftx = fft(Ftx); % compute the DFT

% Smybol mapper block:
symbolMapOp = zeros(L, M); % the o/p of the symbol mapper
% When we map the symbols at the output of the DFT precoder
% to the inputs of the IDFT block, there will be some inputs
% left over. This will be assigned values of "0". We can work
% out the "excess" inputs as (L/2-1) - N.
% We subtract 1 from L/2 as we need to
% leave the first row of the IFFT full of 0s, this is the DC
% subcarrier.
excess = (L/2-1) - N;

switch (mapType)
    case "interleave"
        % The variable below is the amount of zeros that we will insert between
        % each symbol on the input pins of the IDFT block:
        interPadLength = floor(excess/(N-1));
        % Below we are mapping each of the (N) outputs of the DFT block to
        % the (L) inputs of the IDFT block
        for i = 1:N
            symbolMapOp(1+1+(i-1)*(interPadLength+1), :) = Ftx(i, :);
        end
    case "local"
        % insertIndex is the index where the first symbol from the output
        % of the Tx-side DFT block should be mapped to in the Tx-side IDFT
        % block. In "localised" mapping, the data (Ftx in our case) is 
        % usually padded above and below by zeros:
        insertIndex = (excess + 1)/2 + 1 + 1;
        symbolMapOp(insertIndex:insertIndex+N-1, :) = Ftx;
    otherwise
        disp("ERROR: check the value of 'mapType' at the top of the script.")
        return
end

% We need to ensure that the i/p sequence to the IDFT (IFFT) block has
% Hermitian symmetry. This will ensure that the o/p of the block is
% real-valued. In order to impose Hermitian symmetry, we must insert a
% a flipped and conjugated version of the upper half of symbolMapOp into
% its lower half. Also, the first row and the middle row must both be
% loaded with zeros (the first row is the DC subcarrier):
symbolMapOp(L/2 + 2:end, :) = conj(flip(symbolMapOp(2:L/2, :), 1));

% Pass through IFFT block:
ofdmSymbols = ifft(symbolMapOp);

% Add the cyclic prefix:
s = [ofdmSymbols(end-cpLen+1:end, :); ofdmSymbols];

% Pass through the channel. This involves the convolution between the
% channel - which is being modelled by a linear filter (h) - and the signal
% (s):
r = zeros(size(s, 1)+length(h)-1, M);
r_noise = zeros(size(s, 1)+length(h)-1, M);

SNRdB = 28; % channel SNR in dB

for j = 1:M
    r(:, j) = conv(h, s(:, j));
    r_noise(:, j) = awgn(r(:, j), SNRdB, "measured"); % create the noisy signal
end

% Remove the cyclic prefix and the tail caused by the channel delay spread.
% We should do this for each of the "M" OFDM symbols:
r_noise = r_noise(cpLen+1:end, :);
r_noise = r_noise(1:end-(length(h) - 1), :);

% Get the received frame:
Frx = fft(r_noise);
FrxCopy = Frx; % making a copy

% Equalisation stage. First we get the channel response.
channelResponse = fft([h; zeros(L-length(h), 1)]);
% Now we perform the equalisation:
R = Frx./channelResponse;

R = R(2:L/2, :); % extracting the payload

% Rx-side Symbol Mapper:
% Now we must remove the 0s before passing the resulting sequence of
% symbols to the IDFT (IFFT) block:
switch (mapType)
    case "interleave"
        R = R(1:interPadLength+1:end, :);
    case "local"
        R = R(insertIndex+1:insertIndex+N, :);
    otherwise
        disp("ERROR: check the value of 'mappingType' at the top of the script.")
        return
end

% Pass through the Rx-side IFFT block:
R = ifft(R);

R = R(:); % turning R from a matrix to a row or column vector

% Let us plot the constellation:
scatter(real(modulatedData), imag(modulatedData), [50], txSymbolsColour, 'Marker', '*', 'LineWidth', 1.5);
title("\bf\fontname{Georgia}\fontsize{14}Constellation Chart " + "(" + IEEEstandard + " with DFT-spread), K = " + K + ", " + modOrder + "-" + modType + ", SNR = " + SNRdB + " dB, Equalisation at Rx");
ylabel('\bf\fontname{Georgia}\fontsize{12}Quadrature Component');
xlabel('\bf\fontname{Georgia}\fontsize{12}In-phase Component');
set(gca,'Fontname', 'Georgia');
hold on
scatter(real(R(:)), imag(R(:)), [20], rxSymbolsColour1, 'Marker', '+', 'LineWidth', 0.8);
legend({'\bf\fontname{Georgia}\fontsize{14}Tx Symbols', '\bf\fontname{Georgia}\fontsize{14}Rx Symbols'}, 'Location', 'best', 'Orientation', 'vertical');
hold off