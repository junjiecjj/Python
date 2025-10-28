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
% ALSO: this script compares the performance of the periodogram-based
% approach both with and without the DFT-spread stage.

% Setup of variables and system parameters:
N = 64; % no. of complex symbols going into Tx-side DFT block
M = 128; % no. of blocks or symbols being used in OFDM frame
L = 2*N; % size of Tx-side IFFT block

% The boolean variables below control which graphs get generated:
plotNormalHeatMap = true;
plotDFTspreadHeatMap = true;

% General constants:
c = 3e8; % The speed of light

% OFDM system parameters:
To = 4e-6; % The OFDM symbol period
fC = 5.5e9; % The frequency of the centre subcarrier
subcarrierSpacing = 312.5e3; % The subcarrier spacing

% Target parameters:
Vrel = 10; % The relative velocity of our target to the transmitter
d = 25; % The distance between the target and the transmitter
targetRCS = 10; % The radar cross section of the target
targetPhi = 2*pi*rand; % The random phase offset caused by the target

% Working out the approximate Doppler shift caused when the EM wave impinges on the
% target:
fD = 2*fC*(Vrel/c);

% And the time delay:
timeDelay = 2*d/c;

% The attenuation factor, b:
b = sqrt((c*targetRCS)/((4*pi)^3*(d^4)*(fC^2)));
% Info. on modulation:
modType = "QAM"; % type of modulation being used
modOrder = 16; % order of modulation

useHermitianSymmetry = true;

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

% This is where we can form the received frame (Frx):
Frx1 = zeros(L, M); % the first received frame
Frx2 = zeros(N, M); % the second received frame

noisedBW = -100;
Z1 = wgn(L, M, noisedBW, 'complex');
Z2 = wgn(N, M, noisedBW, 'complex');

% Defining the first received frame:
for row = 1:N
    for column = 1:M
        Frx1(row, column) = b*symbolMapOp(row, column)*exp(1i*2*pi*(column - 1)*To*fD)*exp(-1i*2*pi*(row - 1)*timeDelay*subcarrierSpacing) + Z1(row, column);
    end
end

% Defining the second received frame:
for row = 1:N
    for column = 1:M
        Frx2(row, column) = b*FtxCopy(row, column)*exp(1i*2*pi*(column - 1)*To*fD)*exp(-1i*2*pi*(row - 1)*timeDelay*subcarrierSpacing) + Z2(row, column);
    end
end

% Dividing each of the received frames by the transmitted frames:
F1 = Frx1./symbolMapOp;
F2 = Frx2./FtxCopy;
F1(isinf(F1)) = 0; % replace infinity values by 0s

% And now we can remove the rows with "Inf" values, which were caused
% by dividing by "0"s in the transmit frames (symbolMapOp and Ftx)
F1(1, :) = 0;
F1(L/2 + 1, :) = 0;
F2(1, :) = 0;
F2(N/2 + 1, :) = 0;

% Size of FFT and IFFT calculations to be done on the rows and columns of
% the matrix F; these are also the number of rows and columns of the
% uncropped periodogram. Increasing the number of rows and columns in the
% FFT and IFFT calculations increases the resolution, but it also increases
% the number of computations needing to be done...
Nper1 = 4*L;
Mper1 = 4*M;
Nper2 = 4*N;
Mper2 = 4*M;

G = 1/4; % Fraction of OFDM symbol used as a guard interval (TG/T)

% The subcarrier spacing should be at least one order of magnitude
% bigger than the largest occurring Doppler shift - hence a value of 1/10 
% for the quantity D...
D = 1/10;

% Defining our search ranges for each method (the DFT-spread and non
% DFT-spread techniques):
Nmax1 = G*Nper1;
Mmax1 = round(D*Mper1);
Nmax2 = G*Nper2;
Mmax2 = round(D*Mper2);

% Complex periodogram (Cper) calculation:
Cper1 = F1;
% Method 1 to calculate the complex periodogram:
Cper1 = Cper1.';
Cper1 = fft(Cper1, Mper1); % FFT of each row (length of FFT is Mper)
Cper1 = Cper1.';
Cper1 = ifft(Cper1, Nper1); % IFFT of each column...
numCols1 = size(Cper1, 2);
Cper1 = Cper1(1:Nmax1, :);
Cper1 = flip(fftshift(Cper1, 2), 1);
Cper1 = Cper1(:, (numCols1/2)-Mmax1:(numCols1/2)+Mmax1-1);

% Complex periodogram (Cper) calculation:
Cper2 = F2;
% Method 1 to calculate the complex periodogram:
Cper2 = Cper2.';
Cper2 = fft(Cper2, Mper2); % FFT of each row (length of FFT is Mper)
Cper2 = Cper2.';
Cper2 = ifft(Cper2, Nper2); % IFFT of each column...
numCols2 = size(Cper2, 2);
Cper2 = Cper2(1:Nmax2, :);
Cper2 = flip(fftshift(Cper2, 2), 1);
Cper2 = Cper2(:, (numCols2/2)-Mmax2:(numCols2/2)+Mmax2-1);

% First periodogram calculation:
Per1 = 1/(Nmax1*(2*Mmax1 + 1))*(abs(Cper1).^2);

% Second periodogram calculation:
Per2 = 1/(Nmax2*(2*Mmax2 + 1))*(abs(Cper2).^2);

% Getting the maximum values from the periodograms:
maxPer1 = max(Per1(:));
maxPer2 = max(Per2(:));

% Getting the x and y coordinates of any targets that were detected by in
% each case (DFT-spread technique in case 1. And no DFT-spread technique in
% case 2):
[y1, x1] = ind2sub(size(Per1), find(Per1 == maxPer1));
[y2, x2] = ind2sub(size(Per2), find(Per2 == maxPer2));

m_hat1 = x1 - (size(Per1, 2)/2 + 1);
n_hat1 = size(Per1, 1) - y1 + 1;
m_hat2 = x2 - (size(Per2, 2)/2 + 1);
n_hat2 = size(Per2, 1) - y2 + 1;

tau_hat1 = (n_hat1 - 1)/(Nper1*subcarrierSpacing);
distance1 = (n_hat1 - 1)*c/(2*subcarrierSpacing*Nper1);
velocity1 = (m_hat1 - 1)*c/(2*fC*To*Mper1);

tau_hat2 = (n_hat2 - 1)/(Nper2*subcarrierSpacing);
distance2 = (n_hat2 - 1)*c/(2*subcarrierSpacing*Nper2);
velocity2 = (m_hat2 - 1)*c/(2*fC*To*Mper2);

% The index vectors below can be used to find the distance
% and velocity values, which can be used to label the axes...
mIndexes1 = (1:size(Per1, 2)) - (size(Per1, 2)/2 + 1);
nIdexes1 = (1:size(Per1, 1));

mIndexes2 = (1:size(Per2, 2)) - (size(Per2, 2)/2 + 1);
nIdexes2 = (1:size(Per2, 1));

% Working out the corresponding distance and velocity vector
% associated with each (n, m) index...
% Working out the corresponding distance and velocity vector
% associated with each (n, m) index...
distancesVec1 = (nIdexes1 - 1)*c/(2*subcarrierSpacing*Nper1);
velocitiesVec1 = round((mIndexes1 - 1)*c/(2*fC*To*Mper1));
distancesVec2 = (nIdexes2 - 1)*c/(2*subcarrierSpacing*Nper2);
velocitiesVec2 = round((mIndexes2 - 1)*c/(2*fC*To*Mper2));

% Creating label variables to use wwith the two heatmaps below:
distanceLabels1 = string(distancesVec1);
distanceLabels1(~(ceil(distancesVec1) == floor(distancesVec1))) = "";
velocityLabels1 = string(velocitiesVec1);
velocityLabels1(~(mod(velocitiesVec1, 10) == 0)) = "";

distanceLabels2 = string(distancesVec2);
distanceLabels2(~(ceil(distancesVec2) == floor(distancesVec2))) = "";
velocityLabels2 = string(velocitiesVec2);
velocityLabels2(~(mod(velocitiesVec2, 10) == 0)) = "";

if (plotDFTspreadHeatMap == true)
    figure(1)
    h1 = heatmap(velocitiesVec1, flip(distancesVec1), Per1, 'Colormap', jet(200));
    s1 = struct(h1);
    s1.XAxis.TickLabelRotation = 90;
    h1.YDisplayLabels = flip(distanceLabels1);
    h1.XDisplayLabels = velocityLabels1;
    title("\fontsize{14}\fontname{Georgia}Periodogram Method (d = " + d + ", V_{rel.} = " + Vrel + ", f_{D} = " + fD + " Hz" + ", N = " + N + ", M = " + M + ", " + modOrder + "-" + modType + ", G = " + G + ")");
    xlabel('\fontname{Georgia}\bf\itv\rm\bf_{rel.} (m/s)');
    ylabel('\fontname{Georgia}\bfDistance (m)');
end

if (plotNormalHeatMap == true)
    figure(2)
    h2 = heatmap(velocitiesVec2, flip(distancesVec2), Per2, 'Colormap', jet(200));
    s2 = struct(h2);
    s2.XAxis.TickLabelRotation = 90;
    h2.YDisplayLabels = flip(distanceLabels2);
    h2.XDisplayLabels = velocityLabels2;
    title("\fontsize{14}\fontname{Georgia}Periodogram Method (d = " + d + ", V_{rel.} = " + Vrel + ", f_{D} = " + fD + " Hz" + ", N = " + N + ", M = " + M + ", " + modOrder + "-" + modType + ", G = " + G + ")");
    xlabel('\fontname{Georgia}\bf\itv\rm\bf_{rel.} (m/s)');
    ylabel('\fontname{Georgia}\bfDistance (m)');
end