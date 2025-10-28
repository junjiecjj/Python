%% OFDM Communication Link:
% This script models a basic OFDM communication link.
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
mappingType = "interleaving";

% Setting up variables and system parameters:
N = 64; % size of the IFFT block available at Tx
M = 8; % number of OFDM symbols in a transmit frame
% The output symbols of the DFT block are mapped to the inputs of
% the IFFT block. The IFFT block has a greater length, say "L".
% From reading the literature, L is typically an integer multiple of
% N, the length of the DFT. L = KN, where K is an integer:
K = 4; % the factor by which N is multiplied to get "L"
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

% Next comes the IFFT block. It has a length L = KN. K
% is defined at the top of this program. We need to map
% the symbols in DFTop to the inputs of the IFFT block in
% such a way that the output from the IFFT block has a lower PAPR. 
% If we are mapping N complex symbols to KN inputs on the IFFT block
% then there will be (KN-N) or (L-N) empty pins where 0s will 
% be entered. The tricky question is: Where should these 0s be placed?
% How should we map our symbols to the inputs of the IFFT block?
%
% The variable below is for the input (i/p) to the IFFT block.
% I am using "localised" mapping. This is where the symbols are
% mapped to the center of the IFFT block, with zeros padding both ends
% (at the top, and at the bottom):
L = K*N; % the length of the IFFT block

% A matrix to store the i/p columns to the IFFT. Predefining
% this will mean MATLAB will not need to resize variables inside
% a for loop, which takes much longer:
halfIFFTinput = zeros(L/2 - 1, M);

% When we map the symbols at the output of the DFT precoder
% to the inputs of the IDFT block, there will be some inputs
% left over. This will be assigned values of "0". We can work
% out the "excess" inputs as (L-2)/2 - N
% We subtract 2 from L (the size of the IFFT) as we need to
% leave the first row of the IFFT full of 0s, this is the DC
% subcarrier. Likewise, the middle row (L/2) must all be 0s
% so that we have "Hermitian" symmetry. This will help make the
% output of the IFFT block real-valued:
excess = (L-2)/2 - N;

switch mappingType
    case "interleaving"
        for i = 1:N
            halfIFFTinput(2*i - 1, :) = DFToutput(i, :);
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
cpLen = G*L; % the cyclic prefix length; L is the size of the IFFT block

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

ofdmSymbolsReceived = reshape(r, L+cpLen, M);

% Now we can remove the cyclic prefix from each of the symbols:
ofdmSymbolsReceived = ofdmSymbolsReceived(cpLen+1:end, :);


% Now we take the FFT of each symbol... this will form the received frame:
rxFFToutput = fft(ofdmSymbolsReceived); % taking the FFT of each column
rxIFFTinput = rxFFToutput(2:L/2, :); % extracting the payload from the frame...

% Now we must remove the 0s before passing the resulting sequence of
% symbols to the IDFT (IFFT) block:
switch mappingType
    case "interleaving"
        rxIFFTinput = rxIFFTinput(1:2:end, :);
    case "localised"
        insertIndex = (excess + 1)/2;
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