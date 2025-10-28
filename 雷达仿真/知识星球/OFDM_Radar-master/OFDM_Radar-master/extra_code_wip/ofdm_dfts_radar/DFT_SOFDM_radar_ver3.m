%% DFT-spread OFDM (DFT SOFDM) Radar:
% Trying to combine DFT-spread technique with the reciprocal filtering
% approach used in OFDM radar. Some of the variables are the same as those
% used in "Comparison of Correlation-based OFDM Radar Receivers".
%
% Author: Dylan Boland (Student)
%
% NOTE: the correlation is done between the sequence of symbols that is
% loaded into the Tx-side IDFT block and the sequence of symbols that comes
% out the the Rx-side DFT block.

% Setup of variables and system parameters:
N = 32; % no. of complex symbols going into Tx-side DFT block
M = 4; % no. of blocks or symbols being used in OFDM frame
L = 4*N; % size of Tx-side IFFT block

% Info. on modulation:
modType = "QAM"; % type of modulation being used
modOrder = 16; % order of modulation

% Boolean variable to control whether the CP is removed or not before
% processing on the receive side:
removeCP = false;

% Info. on mapping from DFT block to IFFT block (Tx side):
mapType = "local";

switch (modType)
    case "QAM"
        signalSet = qammod((0:modOrder-1), modOrder, "UnitAveragePower", true); % Generate the signal set for the relevant QAM scheme (16, 32, 64 etc.)     
    case "PSK"
        signalSet = pskmod((0:modOrder-1), modOrder);
    otherwise
        disp("Check the status of 'modulationType' above.")
        return % Exit the program early
end

Ftx = zeros(N, M); % a frame to hold our transmit symbols

for subchannel = 1:N % let us load data symbols into our frame
    Ftx(subchannel, :) = randsample(signalSet, M, true); % taking a random sample
end

% Processing and Transmission stage:
FtxCopy = Ftx; % making a copy for receiver-side processing
Ftx = fft(Ftx); % passing the transmit frame to the Tx-side DFT block

% Now we need to map the o/p of DFT block to the i/p of IDFT block:
excess = (L/2-1) - N; % the no. of excess i/p pins on IDFT block
symbolMapOp = zeros(L, M); % output of symbol mapper - this goes into IDFT

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

% Add Cylic Prefix (CP):
G = 1/8; % the fraction of the OFDM symol which is used as a guard interval
cpLen = G*L; % the cyclic prefix (CP) length

% Taking the bottom section of the symbolMapOp matrix and prefixing it to
% the top of the symbolMapOp matrix. This has the effect of adding the CP
% to each column (OFDM symbol):
symbolMapOp = [symbolMapOp(end-cpLen+1:end, :); symbolMapOp];

% Forming the transmit sequence (s). This can be thought of as the o/p of
% the parallel-to-serial block on the Tx side:
s = symbolMapOp(:);

% Target parameters:
a = 2e-3; % how much the signal is attenuated by the target
fD = 100; % the doppler shift caused by the target
delay = 10; % the delay of the received signal (r) in units of samples

% Channel parameters:
noisedBW = -40; % noise level in dBW
Z = wgn(length(s), 1, noisedBW, 'complex'); % complex Gaussian White noise vector...

% Computing the received signal r:
p = 0:(L+cpLen)*M -1; % an index variable that will be used below
dopplerShift = exp(1i*2*pi*fD*p/(L+cpLen)).';
r = [zeros(delay, 1); a*dopplerShift.*s + Z]; % the received signal also has a delay in it

% Extract M blocks for processing. M blocks equals
% M*(L+cpLen) samples, i.e. the length of s:
r = r(1:length(s));
r = reshape(r, L+cpLen, M); % reshaping for block-by-block processing
s = reshape(s, L+cpLen, M); % reshaping for block-by-block processing

if (removeCP == true)
    r = r(cpLen+1:end, :);
    s = s(cpLen+1:end, :);
end

% Receiver-side Processing:
p = 0:size(r, 1)-1; % the vector p, as described in the paper referenced at the top of the program
nu = -800:25:800; % vector of frequencies

% The matrix below will store all the Xm matrices:
Xm_storage = zeros(M*length(nu), 2*size(r, 1)-1);

% The matrix below is the result of multiplying the p vector
% by each value in the nu vector, and storing the results in
% the rows of this "p_nu_matrix" matrix. This is simply being
% used to do the computations compactly.
p_nu_matrix = p.*nu';

% The term below appears at the end of equation 6 in the paper:
negDopplerShift = exp(-1i*2*pi*p_nu_matrix/size(r, 1));

% Preallocating enough memory for this variable which will be used inside
% the for loop below. The number of rows equals the number of subcarriers
% or rows on the Tx-side IDFT (IFFT) block:
r_dash = zeros(size(r, 1), length(nu));

% This for loop works out Xm the reciprocal filtering case:
for block = 1:M
    % sm stands for the part of s which forms OFDM symbol or
    % "block" m. rm stands for the part of r which is considered as the
    % block or symbol m at the receiver:
    sm = 1./conj(s(:, block)); % the signal sm
    sm(isinf(sm)) = 0; % replace "Inf" values with 0 (these will be ignored)
    rm = r(:, block); % the signal rm
    r_dash = rm.*negDopplerShift.';
    
    % Now to calculate the correlation:
    Xm = zeros(length(nu), 2*size(r, 1)-1);
    
    % Computing Xm by using the xcorr() function
    for j = 1:length(nu)
        Xm(j, :) = xcorr(r_dash(:, j), sm);
    end
    
    Xm = (1/sqrt(size(r, 1)))*Xm; % Scaling, as shown in equation 6
    % Now we load this matrix for Xm into the storage matrix so it can be
    % further processed later on...
    Xm_storage(length(nu)*block-length(nu)+1: length(nu)*block, :) = Xm;
end

% By this stage we have Xm calculated for each value of m, where m is the
% index or block number - going from 0 to (M-1). Now we can work out
% equation 5:
negDopplerShift_m = exp(-1i*2*pi*nu'.*(0:M-1)); % term used in equation 5

% This is a matrix to store the overall result of the
% ambiguity function. The values in this matrix will
% be computed using the contents of the Xm matrix.
Xm_dash = zeros(M*length(nu), 2*size(r, 1)-1);

for block = 1:M
    startRowIndex = length(nu)*block - length(nu) + 1;
    endRowIndex = length(nu)*block;
    % The term below (Xm_dash) is the equal to the product of the two terms
    % to the right of the "sum" symbol in equation 5:
    Xm_dash(startRowIndex: endRowIndex, :) = Xm_storage(startRowIndex: endRowIndex, :).*negDopplerShift_m(:, block);
end

% This matrix below will store the final result of the ambiguity function
% computation:
X = zeros(length(nu), 2*size(r, 1)-1);

% Now to implement equation 5:
for block = 1:M
    startRowIndex = length(nu)*block - length(nu) + 1;
    endRowIndex = length(nu)*block;
    X(1:length(nu), :) = X(1:length(nu), :) + Xm_dash(startRowIndex: endRowIndex, :);
end

X = (1/(sqrt(M)))*X; % scaling, as shown in the paper in equation 5

delayVec = -size(r, 1)+1:size(r, 1)-1; % a vector of delays or lags (in units of samples)
delayLabels = string(delayVec);
delayLabels(~(mod(delayVec, 10) == 0)) = ""; % if value in vector is NOT divisble by 10, leave blank
freqLabels = string(nu);
freqLabels(~(mod(nu, 10) == 0)) = ""; % if value in vector is NOT divisble by 10, leave blank

h = heatmap(delayVec, nu, abs(X), 'Colormap', jet(300));
title("\fontsize{14}\fontname{Georgia}Reciprocal Filter with DFT-spread (\tau = " + delay + ", f_{D} = " + fD + " Hz, \sigma^{2} = " + noisedBW + " dBW, " + "N = " + N + ", M = " + M + ", " + modOrder + "-" + modType + ", G = " + G + ")");
xlabel('\fontname{Georgia}\bf Delay (Samples)');
ylabel('\fontname{Georgia}\bf\itf\rm\bf (Hz)');
h.XDisplayLabels = delayLabels;
h.YDisplayLabels = freqLabels;