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
%
% ALSO: this script compares the performance of the reciprocal filter both
% with and without the DFT-spread stage.

% Setup of variables and system parameters:
N = 64; % no. of complex symbols going into Tx-side DFT block
M = 14; % no. of blocks or symbols being used in OFDM frame
L = 4*N; % size of Tx-side IFFT block

% Info. on modulation:
modType = "QAM"; % type of modulation being used
modOrder = 16; % order of modulation

% Boolean variable to control whether the CP is removed or not before
% processing on the receive side:
removeCP = false;
% The boolean variables below control which graphs get generated:
plotNormalHeatMap = true;
plotDFTspreadHeatMap = true;

% The boolean variable below controls whether Hermitian symmetry is used in
% when forming the transmit frame (Ftx) for the non DFT-spread use case:
useHermitianSymmetry = false;
% Info. on mapping from DFT block to IFFT block (Tx side):
mapType = "interleave";

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

if (useHermitianSymmetry == true)
    for subchannel = 2:N/2
        Ftx(subchannel, :) = randsample(signalSet, M, true); % taking a random sample
    end
    Ftx = [Ftx(1:N/2, :); zeros(1, M); conj(flip(Ftx(2:N/2, :), 1))];
else
    for subchannel = 1:N % let us load data symbols into our frame
        Ftx(subchannel, :) = randsample(signalSet, M, true); % taking a random sample
    end
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
G = 0; % the fraction of the OFDM symol which is used as a guard interval
cpLen = G*L; % the cyclic prefix (CP) length

% Taking the bottom section of the symbolMapOp matrix and prefixing it to
% the top of the symbolMapOp matrix. This has the effect of adding the CP
% to each column (OFDM symbol):
symbolMapOp = [symbolMapOp(end-cpLen+1:end, :); symbolMapOp];
FtxCopy = [FtxCopy(end-cpLen+1:end, :); FtxCopy]; % used in forming s2

% Forming our two transmit sequences. s1 is the transmit sequence resulting
% from the use of the DFT-spread technique. s2 is the transmit sequence
% resulting from the standard transmission scheme, without the DFT-spread
% technique being used:
s1 = symbolMapOp(:);
s2 = FtxCopy(:);

% Let us also workout the peak-to-average power in each case:
PAPR1 = max(abs(s1).^2)/mean(abs(s1).^2); % PAPR when DFT spread is used
PAPR2 = max(abs(s2).^2)/mean(abs(s2).^2); % PAPR when no DFT-spread is used

% Target parameters:
a = 2e-3; % how much the signal is attenuated by the target
fD = 100; % the doppler shift caused by the target
delay = 10; % the delay of the received signal (r) in units of samples

% Channel parameters:
noisedBW = -25; % noise level in dBW
% We have two noise vectors: one for s1, and another for s2.
Z1 = wgn(length(s1), 1, noisedBW, 'complex');
Z2 = wgn(length(s2), 1, noisedBW, 'complex');

% Computing the received signal r:
p1 = 0:(L+cpLen)*M - 1; % an index variable used working out r1
p2 = 0:(N+cpLen)*M - 1; % another used for working out r2
dopplerShift1 = exp(1i*2*pi*fD*p1/(L+cpLen)).'; % for r1 expression
dopplerShift2 = exp(1i*2*pi*fD*p2/(N+cpLen)).'; % for r2 expression
r1 = [zeros(delay, 1); a*dopplerShift1.*s1 + Z1]; % defining r1
r2 = [zeros(delay, 1); a*dopplerShift2.*s2 + Z2]; % defining r2

% At the receiver, there will be a section of the receive sequence which
% will be unused during the correlation process.
r1 = r1(1:length(s1));
r2 = r2(1:length(s2));
% Reshaping the transmit and receive sequences in order to allow for more
% easier block-by-block processing:
r1 = reshape(r1, L+cpLen, M);
s1 = reshape(s1, L+cpLen, M);
r2 = reshape(r2, N+cpLen, M);
s2 = reshape(s2, N+cpLen, M);

% Some signal processing techniques are based on retaining the cyclic
% prefix (CP). Others require the CP to be removed. With this in mind, 
% we will make it easy to decide whether the CP is to be removed or not:
if (removeCP == true) % removeCP is a boolean variable defined at the top
    r1 = r1(cpLen+1:end, :);
    s1 = s1(cpLen+1:end, :);
    r2 = r2(cpLen+1:end, :);
    s2 = s2(cpLen+1:end, :);
end

% Receiver-side Processing. We will have two index vectors: p1 and p2
p1 = 0:size(r1, 1)-1;
p2 = 0:size(r2, 1)-1;
nu = -400:25:400; % vector of frequencies (this is correlation stage)

% Xm_storage1 is DFT-spread analysis. Xm_storage2 is for non DFT-spread
% analysis:
Xm_storage1 = zeros(M*length(nu), 2*size(r1, 1)-1);
Xm_storage2 = zeros(M*length(nu), 2*size(r2, 1)-1);
% The matrix below is the result of multiplying the p vector
% by each value in the nu vector, and storing the results in
% the rows of this "p_nu_matrix" matrix. This is simply being
% used to do the computations compactly.
p_nu_matrix1 = p1.*nu';
p_nu_matrix2 = p2.*nu';

% The term below appears at the end of equation 6 in the paper:
negDopplerShift1 = exp(-1i*2*pi*p_nu_matrix1/size(r1, 1));
negDopplerShift2 = exp(-1i*2*pi*p_nu_matrix2/size(r2, 1));

r_dash1 = zeros(size(r1, 1), length(nu));
r_dash2 = zeros(size(r2, 1), length(nu));

% This for loop works out Xm the reciprocal filtering case:
for block = 1:M
    % The correlation between the transmit and receive sequences is done
    % here. sm1 represents the mth symbol (or block) of the transmitted
    % sequence s1:
    sm1 = 1./conj(s1(:, block));
    sm2 = 1./conj(s2(:, block));
    sm1(isinf(sm1)) = 0;
    sm2(isinf(sm2)) = 0;
    rm1 = r1(:, block);
    rm2 = r2(:, block);
    r_dash1 = rm1.*negDopplerShift1.';
    r_dash2 = rm2.*negDopplerShift2.';
    
    % The variables below store the correlation between the mth symbols or
    % blocks.
    Xm1 = zeros(length(nu), 2*size(r1, 1)-1);
    Xm2 = zeros(length(nu), 2*size(r2, 1)-1);
    
    % Computing Xm by using the xcorr() function
    for j = 1:length(nu)
        Xm1(j, :) = xcorr(r_dash1(:, j), sm1);
        Xm2(j, :) = xcorr(r_dash2(:, j), sm2);
    end
    
    % Scaling, as shown in equation 6
    Xm1 = (1/sqrt(size(r1, 1)))*Xm1;
    Xm2 = (1/sqrt(size(r2, 1)))*Xm2;
    % Now we load this matrix for Xm into the storage matrix so it can be
    % further processed later on...
    Xm_storage1(length(nu)*block-length(nu)+1: length(nu)*block, :) = Xm1;
    Xm_storage2(length(nu)*block-length(nu)+1: length(nu)*block, :) = Xm2;
end

negDopplerShift_m = exp(-1i*2*pi*nu'.*(0:M-1)); % term used in equation 5

% These matrix variables stores the results of multiplying each Xm matrix
% by the negative Doppler shift term defined just above:
Xm_dash1 = zeros(M*length(nu), 2*size(r1, 1)-1);
Xm_dash2 = zeros(M*length(nu), 2*size(r2, 1)-1);

for block = 1:M
    startRowIndex = length(nu)*block - length(nu) + 1;
    endRowIndex = length(nu)*block;
    % The term below (Xm_dash) is the equal to the product of the two terms
    % to the right of the "sum" symbol in equation 5:
    Xm_dash1(startRowIndex: endRowIndex, :) = Xm_storage1(startRowIndex: endRowIndex, :).*negDopplerShift_m(:, block);
    Xm_dash2(startRowIndex: endRowIndex, :) = Xm_storage2(startRowIndex: endRowIndex, :).*negDopplerShift_m(:, block);
end

% This matrix below will store the final result of the ambiguity function
% computation:
X1 = zeros(length(nu), 2*size(r1, 1)-1);
X2 = zeros(length(nu), 2*size(r2, 1)-1);

% Now to implement equation 5:
for block = 1:M
    startRowIndex = length(nu)*block - length(nu) + 1;
    endRowIndex = length(nu)*block;
    X1(1:length(nu), :) = X1(1:length(nu), :) + Xm_dash1(startRowIndex: endRowIndex, :);
    X2(1:length(nu), :) = X2(1:length(nu), :) + Xm_dash2(startRowIndex: endRowIndex, :);
end

% Scaling as shown in equation 5 of the paper:
X1 = (1/(sqrt(M)))*X1;
X2 = (1/(sqrt(M)))*X2;

delayVec1 = -size(r1, 1)+1:size(r1, 1)-1;
delayVec2 = -size(r2, 1)+1:size(r2, 1)-1;
delayLabels1 = string(delayVec1);
delayLabels2 = string(delayVec2);
delayLabels1(~(mod(delayVec1, 10) == 0)) = ""; % if value in vector is NOT divisble by 10, leave blank
delayLabels2(~(mod(delayVec2, 10) == 0)) = ""; % if value in vector is NOT divisble by 10, leave blank
freqLabels = string(nu);
freqLabels(~(mod(nu, 10) == 0)) = ""; % if value in vector is NOT divisble by 10, leave blank

% Next we can plot the results. plotNormalHeatMap and plotDFTspreadHeatMap
% are boolean variables defined at the top of the program:
if (plotDFTspreadHeatMap)
    figure(1)
    h1 = heatmap(delayVec1, nu, abs(X1), 'Colormap', jet(300));
    title("\fontsize{14}\fontname{Georgia}Reciprocal Filter with DFT-spread (\tau = " + delay + ", f_{D} = " + fD + " Hz, \sigma^{2} = " + noisedBW + " dBW, " + "N = " + N + ", M = " + M + ", " + modOrder + "-" + modType + ", G = " + G + ")");
    xlabel('\fontname{Georgia}\bf Delay (Samples)');
    ylabel('\fontname{Georgia}\bf\itf\rm\bf (Hz)');
    h1.XDisplayLabels = delayLabels1;
    h1.YDisplayLabels = freqLabels;
end

if (plotNormalHeatMap)
    figure(2)
    h2 = heatmap(delayVec2, nu, abs(X2), 'Colormap', jet(300));
    title("\fontsize{14}\fontname{Georgia}Reciprocal Filter (\tau = " + delay + ", f_{D} = " + fD + " Hz, \sigma^{2} = " + noisedBW + " dBW, " + "N = " + N + ", M = " + M + ", " + modOrder + "-" + modType + ", G = " + G + ")");
    xlabel('\fontname{Georgia}\bf Delay (Samples)');
    ylabel('\fontname{Georgia}\bf\itf\rm\bf (Hz)');
    h2.XDisplayLabels = delayLabels2;
    h2.YDisplayLabels = freqLabels;
end