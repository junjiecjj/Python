%% OFDM Communication Link using DFT-spread:
% This script models the case of DFT-spread OFDM (DFT SOFDM).
% The DFT-spread technique has been investigated as a way to combat
% the high PAPR of OFDM transmissions. A high PAPR puts high demands
% on the linear power amplifier at the transmitter. For uplink transmission
% we want the power consumption in the user equipment (UE) to be low. Hence
% a high PAPR is not desirable. As a result, DFT-spread OFDM has been
% adopted as the uplink modulation scheme for LTE 3G systems, as it reduces
% the PAPR and therefore improves power efficiency. This script also
% investigates the use of DFT-spread in OFDM radar.
%
% Author: Dylan Boland (Student)

% The boolean variable below controls whether we are simulating the
% case where the OFDM system is being used for communication or for
% radar:
usingAsRadar = false;

% This boolean variable controls whether unit average power during
% the modulation phase is needed or not:
unitAvgPower = true;

% This variable controls what type of
% mapping from the DFT to IDFT block occurs.
% There are two main types covered by this
% program: interleaving and  localised.
mappingType = "localised";

% Setting up variables and system parameters:
N = 64; % size of the IFFT block available at Tx
M = 16; % number of OFDM symbols in a transmit frame
% The output symbols of the DFT block are mapped to the inputs of
% the IFFT block. The IFFT block has a greater length, say "L".
% From reading the literature, L is typically an integer multiple of
% N, the length of the DFT. L = KN, where K is an integer:
K = 2; % the factor by which N is multiplied to get "L"; should be at least 2
modOrder = 16; % order of the modulation scheme being used
bitsPerSymbol = log2(modOrder); % the number of bits each symbol will represent...
numBits = (N)*(M)*(bitsPerSymbol); % the amount (or length) of the data we should generate...
txBits = randi([0 1], numBits, 1); % our transmit bit stream
constellationValues = (0:modOrder-1)'; % the values of the constellation (in decimal)
modulationType = "QAM"; % modulation scheme used; this can be easily changed...

% creating the sequence of transmit symbols:
switch(modulationType)
    case "QAM"
        if (unitAvgPower == true)
            modulatedData = qammod(txBits, modOrder, "InputType", "bit", "UnitAveragePower", true);
        else
            modulatedData = qammod(txBits, modOrder, "InputType", "bit");     
        end
    otherwise
        disp("Check the status of 'modulationType' above.")
        return % exit the program early
end

% Now to form the transmit frame:
Ftx = reshape(modulatedData, N, M);

% Let us work out the PAPR of the complex symbols in the transmit frame:
PAPR1 = max(abs(Ftx(:)))/mean(abs(Ftx(:)));

% Now we should pass each column of modulation symbols
% through the DFT (FFT) block:
DFToutput = fft(Ftx); % taking the DFT of each column...

% Let us work out the PAPR of the complex symbols at the output of DFT:
PAPR2 = max(abs(DFToutput(:)))/mean(abs(DFToutput(:)));

L = K*N; % the length of half of the IFFT block

% A matrix to store the i/p columns to the IFFT. Predefining
% this will mean MATLAB will not need to resize variables inside
% a for loop, which takes much longer:
halfIFFTinput = zeros(L - 1, M);

% When we map the symbols at the output of the DFT precoder
% to the inputs of the IDFT block, there will be some inputs
% left over. This will be assigned values of "0". We can work
% out the "excess" inputs as (L-1) - N.
% We subtract 1 from L (half the size of the IFFT) as we need to
% leave the first row of the IFFT full of 0s, this is the DC
% subcarrier.
excess = (L-1) - N;

switch mappingType
    case "interleaving"
        % The variable below is the amount of zeros that we will insert between
        % each symbol on the input pins of the IFFT:
        interPadLength = floor(excess/(N-1));
        for i = 1:N
            halfIFFTinput(1+(i-1)*(interPadLength+1), :) = DFToutput(i, :);
        end
    case "localised"
        insertIndex = (excess + 1)/2;
        halfIFFTinput(insertIndex+1:insertIndex+N, :) = DFToutput;
    otherwise
        disp("ERROR: check the value of 'mappingType' at the top of the script.")
        return
end

% Imposing Hermitian symmetry so that the output of the IFFT block will be
% real-valued:
IFFTinput = [zeros(1, M); halfIFFTinput; zeros(1, M); conj(flip(halfIFFTinput, 1))];
% Now we pass the sequence into the IDFT (IFFT) block to form the OFDM
% symbol:
ofdmSymbols = ifft(IFFTinput);

% Add the cyclic prefix:
G = 0; % the fraction of a symbol that will be used as the cyclic prefix...
cpLen = G*2*L; % the cyclic prefix length; L is the size of the IFFT block

ofdmSymbols = [ofdmSymbols(end-cpLen+1:end, :); ofdmSymbols]; % adding the CP to each symbol

% Now to form the transmit sequence (s). This is got by
% concatenating all the OFDM symbols together:
s = ofdmSymbols(:); % the transmit sequence (s)

% Let us work out the PAPR of the OFDM signal (s):
PAPR3 = max(abs(s))/mean(abs(s));

channelSNR = 20; % the channel's signal-to-noise ratio...

if (usingAsRadar == false)
    r = awgn(s, channelSNR, "measured");
end

ofdmSymbolsReceived = reshape(r, 2*L+cpLen, M);

% Now we can remove the cyclic prefix from each of the symbols:
ofdmSymbolsReceived = ofdmSymbolsReceived(cpLen+1:end, :);


% Now we take the FFT of each symbol... this will form the received frame:
rxFFToutput = fft(ofdmSymbolsReceived); % taking the FFT of each column
rxIFFTinput = rxFFToutput(2:L, :); % extracting the payload from the frame...

% Now we must remove the 0s before passing the resulting sequence of
% symbols to the IDFT (IFFT) block:
switch mappingType
    case "interleaving"
        rxIFFTinput = rxIFFTinput(1:interPadLength+1:end, :);
    case "localised"
        rxIFFTinput = rxIFFTinput(insertIndex+1:insertIndex+N, :);
    otherwise
        disp("ERROR: check the value of 'mappingType' at the top of the script.")
        return
end

% The receiver's IDFT block, to mirror the DFT block on the transmitter's
% side:
rxIFFToutput = ifft(rxIFFTinput);

% Now we must demodulate the received symbols in order to get back the
% bits:
if (unitAvgPower == true)
    rxBits = qamdemod(rxIFFToutput, modOrder, "OutputType", "bit", "UnitAveragePower", true);
else
    rxBits = qamdemod(rxIFFToutput(:), modOrder, "OutputType", "bit");
end

% We can work out the number of the bit errors by comparing the bit stream
% that was sent out and the bit stream that was received. When two vectors
% are compared using the "==" operator, the result is a "logical" vector
% made up of 1s and 0s. We can use this idea to compare the received bits
% (rxBits) with the transmitted ones (txBits):
rxBits = rxBits(:); % converting to a bit stream
numBitErrors = numBits - sum(rxBits == txBits);

BER = numBitErrors/numBits; % the bit error rate (BER)