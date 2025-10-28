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
N = 52; % no. of complex symbols going into Tx-side DFT block
M = 14; % no. of blocks or symbols being used in OFDM frame
L = 4*N; % size of Tx-side IFFT block

% Info. on modulation:
modType = "QAM"; % type of modulation being used
modOrder = 16; % order of modulation

% Boolean variable to control whether the CP is removed or not before
% processing on the receive side:
removeCP = false;

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

signalPower = mean(abs(s).^2); % average signal power

% Target parameters:
a = 6e-3; % how much the signal is attenuated by the target
fD = 250; % the doppler shift caused by the target
delay1 = 6; % the delay of the received signal (r) in units of samples
delay2 = 6; % delay associated with second path
delay3 = 7;
delay4 = 7;
delay5 = 7;
delay6 = 7;

% Channel parameters:
noisedBW = -30; % noise level in dBW
Z = wgn(length(s), 1, noisedBW, 'complex'); % complex Gaussian White noise vector...

% Computing the received signal r:
p = 0:(L+cpLen)*M -1; % an index variable that will be used below
dopplerShift = exp(1i*2*pi*fD*p/(L+cpLen)).';
r1 = [zeros(delay1, 1); a*dopplerShift.*s + Z]; % the received signal also has a delay in it
r2 = [zeros(delay2, 1); a*dopplerShift.*s + Z];
r3 = [zeros(delay3, 1); a*dopplerShift.*s + Z];
r4 = [zeros(delay4, 1); a*dopplerShift.*s + Z];
r5 = [zeros(delay5, 1); a*dopplerShift.*s + Z];
r6 = [zeros(delay6, 1); a*dopplerShift.*s + Z];

% Extract M blocks for processing. M blocks equals
% M*(L+cpLen) samples, i.e. the length of s:
r1 = r1(1:length(s));
r2 = r2(1:length(s));
r3 = r3(1:length(s));
r4 = r4(1:length(s));
r5 = r5(1:length(s));
r6 = r6(1:length(s));
r1 = reshape(r1, L+cpLen, M); % reshaping for block-by-block processing
r2 = reshape(r2, L+cpLen, M); % reshaping for block-by-block processing
r3 = reshape(r3, L+cpLen, M); % reshaping for block-by-block processing
r4 = reshape(r4, L+cpLen, M); % reshaping for block-by-block processing
r5 = reshape(r5, L+cpLen, M); % reshaping for block-by-block processing
r6 = reshape(r6, L+cpLen, M); % reshaping for block-by-block processing
s = reshape(s, L+cpLen, M); % reshaping for block-by-block processing

if (removeCP == true)
    r1 = r1(cpLen+1:end, :);
    r2 = r2(cpLen+1:end, :);
    r3 = r3(cpLen+1:end, :);
    r4 = r4(cpLen+1:end, :);
    r5 = r5(cpLen+1:end, :);
    r6 = r6(cpLen+1:end, :);
    s = s(cpLen+1:end, :);
end

% Receiver-side Processing:
p = 0:size(r1, 1)-1; % the vector p, as described in the paper referenced at the top of the program
nu = -300:25:300; % vector of frequencies

% The matrix below will store all the Xm matrices:
Xm_storage1 = zeros(M*length(nu), 2*size(r1, 1)-1);
Xm_storage2 = zeros(M*length(nu), 2*size(r2, 1)-1);
Xm_storage3 = zeros(M*length(nu), 2*size(r3, 1)-1);
Xm_storage4 = zeros(M*length(nu), 2*size(r4, 1)-1);
Xm_storage5 = zeros(M*length(nu), 2*size(r5, 1)-1);
Xm_storage6 = zeros(M*length(nu), 2*size(r6, 1)-1);

% The matrix below is the result of multiplying the p vector
% by each value in the nu vector, and storing the results in
% the rows of this "p_nu_matrix" matrix. This is simply being
% used to do the computations compactly.
p_nu_matrix = p.*nu';

% The term below appears at the end of equation 6 in the paper:
negDopplerShift = exp(-1i*2*pi*p_nu_matrix/size(r1, 1));

% Preallocating enough memory for this variable which will be used inside
% the for loop below. The number of rows equals the number of subcarriers
% or rows on the Tx-side IDFT (IFFT) block:
r_dash1 = zeros(size(r1, 1), length(nu));
r_dash2 = zeros(size(r2, 1), length(nu));
r_dash3 = zeros(size(r3, 1), length(nu));
r_dash4 = zeros(size(r4, 1), length(nu));
r_dash5 = zeros(size(r5, 1), length(nu));
r_dash6 = zeros(size(r6, 1), length(nu));

% This for loop works out Xm the reciprocal filtering case:
for block = 1:M
    % sm stands for the part of s which forms OFDM symbol or
    % "block" m. rm stands for the part of r which is considered as the
    % block or symbol m at the receiver:
    sm = 1./conj(s(:, block)); % the signal sm
    sm(isinf(sm)) = 0; % replace "Inf" values with 0 (these will be ignored)
    rm1 = r1(:, block); % the signal rm
    rm2 = r2(:, block); % the signal rm
    rm3 = r3(:, block); % the signal rm
    rm4 = r4(:, block); % the signal rm
    rm5 = r5(:, block); % the signal rm
    rm6 = r6(:, block); % the signal rm
    r_dash1 = rm1.*negDopplerShift.';
    r_dash2 = rm2.*negDopplerShift.';
    r_dash3 = rm3.*negDopplerShift.';
    r_dash4 = rm4.*negDopplerShift.';
    r_dash5 = rm5.*negDopplerShift.';
    r_dash6 = rm6.*negDopplerShift.';
    
    % Now to calculate the correlation:
    Xm1 = zeros(length(nu), 2*size(r1, 1)-1);
    Xm2 = zeros(length(nu), 2*size(r2, 1)-1);
    Xm3 = zeros(length(nu), 2*size(r3, 1)-1);
    Xm4 = zeros(length(nu), 2*size(r4, 1)-1);
    Xm5 = zeros(length(nu), 2*size(r5, 1)-1);
    Xm6 = zeros(length(nu), 2*size(r6, 1)-1);
    
    % Computing Xm by using the xcorr() function
    for j = 1:length(nu)
        Xm1(j, :) = xcorr(r_dash1(:, j), sm);
        Xm2(j, :) = xcorr(r_dash2(:, j), sm);
        Xm3(j, :) = xcorr(r_dash3(:, j), sm);
        Xm4(j, :) = xcorr(r_dash4(:, j), sm);
        Xm5(j, :) = xcorr(r_dash5(:, j), sm);
        Xm6(j, :) = xcorr(r_dash6(:, j), sm);
    end
    
    Xm1 = (1/sqrt(size(r1, 1)))*Xm1; % Scaling, as shown in equation 6
    Xm2 = (1/sqrt(size(r2, 1)))*Xm2; % Scaling, as shown in equation 6
    Xm3 = (1/sqrt(size(r3, 1)))*Xm3; % Scaling, as shown in equation 6
    Xm4 = (1/sqrt(size(r4, 1)))*Xm4; % Scaling, as shown in equation 6
    Xm5 = (1/sqrt(size(r5, 1)))*Xm5; % Scaling, as shown in equation 6
    Xm6 = (1/sqrt(size(r6, 1)))*Xm6; % Scaling, as shown in equation 6
    
    % Now we load this matrix for Xm into the storage matrix so it can be
    % further processed later on...
    Xm_storage1(length(nu)*block-length(nu)+1: length(nu)*block, :) = Xm1;
    Xm_storage2(length(nu)*block-length(nu)+1: length(nu)*block, :) = Xm2;
    Xm_storage3(length(nu)*block-length(nu)+1: length(nu)*block, :) = Xm3;
    Xm_storage4(length(nu)*block-length(nu)+1: length(nu)*block, :) = Xm4;
    Xm_storage5(length(nu)*block-length(nu)+1: length(nu)*block, :) = Xm5;
    Xm_storage6(length(nu)*block-length(nu)+1: length(nu)*block, :) = Xm6;
end

% By this stage we have Xm calculated for each value of m, where m is the
% index or block number - going from 0 to (M-1). Now we can work out
% equation 5:
negDopplerShift_m = exp(-1i*2*pi*nu'.*(0:M-1)); % term used in equation 5

% This is a matrix to store the overall result of the
% ambiguity function. The values in this matrix will
% be computed using the contents of the Xm matrix.
Xm_dash1 = zeros(M*length(nu), 2*size(r1, 1)-1);
Xm_dash2 = zeros(M*length(nu), 2*size(r2, 1)-1);
Xm_dash3 = zeros(M*length(nu), 2*size(r3, 1)-1);
Xm_dash4 = zeros(M*length(nu), 2*size(r4, 1)-1);
Xm_dash5 = zeros(M*length(nu), 2*size(r5, 1)-1);
Xm_dash6 = zeros(M*length(nu), 2*size(r6, 1)-1);

for block = 1:M
    startRowIndex = length(nu)*block - length(nu) + 1;
    endRowIndex = length(nu)*block;
    % The term below (Xm_dash) is the equal to the product of the two terms
    % to the right of the "sum" symbol in equation 5:
    Xm_dash1(startRowIndex: endRowIndex, :) = Xm_storage1(startRowIndex: endRowIndex, :).*negDopplerShift_m(:, block);
    Xm_dash2(startRowIndex: endRowIndex, :) = Xm_storage2(startRowIndex: endRowIndex, :).*negDopplerShift_m(:, block);
    Xm_dash3(startRowIndex: endRowIndex, :) = Xm_storage3(startRowIndex: endRowIndex, :).*negDopplerShift_m(:, block);
    Xm_dash4(startRowIndex: endRowIndex, :) = Xm_storage4(startRowIndex: endRowIndex, :).*negDopplerShift_m(:, block);
    Xm_dash5(startRowIndex: endRowIndex, :) = Xm_storage5(startRowIndex: endRowIndex, :).*negDopplerShift_m(:, block);
    Xm_dash6(startRowIndex: endRowIndex, :) = Xm_storage6(startRowIndex: endRowIndex, :).*negDopplerShift_m(:, block);
end

% This matrix below will store the final result of the ambiguity function
% computation:
X1 = zeros(length(nu), 2*size(r1, 1)-1);
X2 = zeros(length(nu), 2*size(r2, 1)-1);
X3 = zeros(length(nu), 2*size(r3, 1)-1);
X4 = zeros(length(nu), 2*size(r4, 1)-1);
X5 = zeros(length(nu), 2*size(r5, 1)-1);
X6 = zeros(length(nu), 2*size(r6, 1)-1);

% Now to implement equation 5:
for block = 1:M
    startRowIndex = length(nu)*block - length(nu) + 1;
    endRowIndex = length(nu)*block;
    X1(1:length(nu), :) = X1(1:length(nu), :) + Xm_dash1(startRowIndex: endRowIndex, :);
    X2(1:length(nu), :) = X2(1:length(nu), :) + Xm_dash2(startRowIndex: endRowIndex, :);
    X3(1:length(nu), :) = X3(1:length(nu), :) + Xm_dash3(startRowIndex: endRowIndex, :);
    X4(1:length(nu), :) = X4(1:length(nu), :) + Xm_dash4(startRowIndex: endRowIndex, :);
    X5(1:length(nu), :) = X5(1:length(nu), :) + Xm_dash5(startRowIndex: endRowIndex, :);
    X6(1:length(nu), :) = X6(1:length(nu), :) + Xm_dash6(startRowIndex: endRowIndex, :);
end

X1 = (1/(sqrt(M)))*X1; % scaling, as shown in the paper in equation 5
X2 = (1/(sqrt(M)))*X2; % scaling, as shown in the paper in equation 5
X3 = (1/(sqrt(M)))*X3; % scaling, as shown in the paper in equation 5
X4 = (1/(sqrt(M)))*X4; % scaling, as shown in the paper in equation 5
X5 = (1/(sqrt(M)))*X5; % scaling, as shown in the paper in equation 5
X6 = (1/(sqrt(M)))*X6; % scaling, as shown in the paper in equation 5

X7 = X1 + X2 + X3 + X4;
delayVec = -size(r1, 1)+1:size(r1, 1)-1; % a vector of delays or lags (in units of samples)
delayLabels = string(delayVec);
delayLabels(~(mod(delayVec, 10) == 0)) = ""; % if value in vector is NOT divisble by 10, leave blank
freqLabels = string(nu);
freqLabels(~(mod(nu, 10) == 0)) = ""; % if value in vector is NOT divisble by 10, leave blank

SNR = 10*log10(a^2/(10^(noisedBW/10))^2)

h = heatmap(delayVec, nu, abs(X7), 'Colormap', jet(300));
%title("\fontsize{14}\fontname{Georgia}Reciprocal Filter with DFT-spread (\tau = " + delay + ", f_{D} = " + fD + " Hz, \sigma^{2} = " + noisedBW + " dBW, " + "N = " + N + ", M = " + M + ", " + modOrder + "-" + modType + ", G = " + G + ")");
xlabel('\fontname{Georgia}\bf Delay (Samples)');
ylabel('\fontname{Georgia}\bf\itf\rm\bf (Hz)');
set(gca,'Fontname', 'Georgia');
h.XDisplayLabels = delayLabels;
h.YDisplayLabels = freqLabels;