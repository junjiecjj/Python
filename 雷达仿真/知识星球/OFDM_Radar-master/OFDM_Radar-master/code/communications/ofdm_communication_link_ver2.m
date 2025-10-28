%% OFDM Communication Link:
% This script models a basic OFDM communication link.
% Author: Dylan Boland (Student)

% The boolean variable below controls whether we are simulating the
% case where the OFDM system is being used for communication or for
% radar:
usingAsRadar = false;

% Setting up variables and system parameters:
N = 64; % size of the IFFT block available at Tx
M = 68; % number of OFDM symbols in a transmit frame
modOrder = 16; % order of the modulation scheme being used
bitsPerSymbol = log2(modOrder); % the number of bits each symbol will represent...
numBits = (N/2 - 1)*(M)*(bitsPerSymbol); % the amount (or length) of the data we should generate...
txBits = randi([0 1], numBits, 1); % our data bit stream
constellationValues = (0:modOrder-1)'; % the value of the constellation (in decimal)
modulationType = "QAM"; % this can be easily changed...

% creating the sequence of transmit symbols:
switch(modulationType)
    case "QAM"
        modulatedData = qammod(txBits, modOrder, "InputType", "bit", "UnitAveragePower", true);     
    otherwise
        disp("Check the status of 'modulationType' above.")
        return % exit the program early
end

Ftx = zeros(N, M); % the transmit frame (empty at the moment)

% Now to form the transmit frame:
Ftx(2:N/2, :) = reshape(modulatedData, N/2-1, M);
Ftx(N/2+2:end, :) = conj(flip(Ftx(2:N/2, :), 1));

G = 0; % the fraction of a symbol that will be used as the cyclic prefix...
cpLen = G*N; % the cyclic prefix length...

ofdmSymbols = ifft(Ftx); % taking the IFFT of each column of Ftx...
ofdmSymbols = [ofdmSymbols(end-cpLen+1:end, :); ofdmSymbols]; % adding the CP to each symbol

% Now to form the transmit sequence (s). This is got by
% concatenating all the OFDM symbols together:
s = ofdmSymbols(:); % the transmit sequence (s)

channelSNR = 0; % the channel's signal-to-noise ratio...

if (usingAsRadar == false)
    r = awgn(s, channelSNR, "measured");
end

ofdmSymbolsReceived = reshape(r, N+cpLen, M);

% Now we can remove the cyclic prefix from each of the symbols:
ofdmSymbolsReceived = ofdmSymbolsReceived(cpLen+1:end, :);

% Now we take the FFT of each symbol... this will form the received frame:
Frx = fft(ofdmSymbolsReceived);

payloadData = Frx(2:N/2, :); % extracting the payload from the frame...

% Now we must demodulate the received symbols in order to get back the
% bits:
rxBits = qamdemod(payloadData(:), modOrder, "OutputType", "bit", "UnitAveragePower", true);

% We can work out the number of the bit errors by comparing the bit stream
% that was sent out and the bit stream that was received. When two vectors
% are compared using the "==" operator, the result is a "logical" vector
% made up of 1s and 0s. We can use this idea to compare the received bits
% (rxBits) with the transmitted ones (txBits):
numBitErrors = numBits - sum(rxBits == txBits);

BER = numBitErrors/numBits; % the bit error rate (BER)