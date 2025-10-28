%% Multiple Target Estimation with Periodogram:
% Author: Dylan Boland (Student)

IEEEstandard = "802.11p"; % the IEEE standard that we want to simulate...

modOrder = 16; % modulation order being used
modType = "QAM"; % type of modulation being used

c = 3e8; % speed of light

% Let us define the OFDM system parameters based on the IEEE 802.11 standard
% in use:
switch (IEEEstandard)
    case "802.11a"
        subcarrierSpacing = 312.5e3; % subcarrier spacing for 802.11a
        Tu = 1/subcarrierSpacing; % the "useful" symbol period
        G = 1/8; % the guard interval fraction
        Tg = G*Tu; % the guard interval time
        To = Tu + Tg; % the OFDM symbol duration (period)
        fC = 5.5e9; % 802.11a uses the 5 GHz band
        N = 52; % 48 for data, and 4 pilot tones
        numPilotTones = 4; % no. pilot tones used 
    case "802.11g"
        subcarrierSpacing = 312.5e3; % subcarrier spacing for 802.11g
        Tu = 1/subcarrierSpacing; % the "useful" symbol period
        G = 1/8; % the guard interval fraction
        Tg = G*Tu; % the guard interval time
        To = Tu + Tg; % the OFDM symbol duration (period)
        fC = 2.4e9; % 802.11a uses the 2.4 GHz band
        N = 52; % 48 for data, and 4 pilot tones
        numPilotTones = 4; % no. pilot tones used
    case "802.11p"
        subcarrierSpacing = 78.125e3; % subcarrier spacing for 802.11g
        Tu = 1/subcarrierSpacing; % the "useful" symbol period
        G = 1/8; % the guard interval fraction
        Tg = G*Tu; % the guard interval time
        To = Tu + Tg; % the OFDM symbol duration (period)
        fC = 5.9e9; % 802.11a uses the 2.4 GHz band
        N = 52; % 48 for data, and 4 pilot tones
        numPilotTones = 4; % no. pilot tones used
end

switch(modType)
    % generating the set of transmit symbols for the modulation scheme
    % that was chosen
    case "QAM"
        modulationAlphabet = qammod((0:modOrder-1), modOrder, 'UnitAveragePower', true);
    case "QPSK"
        modulationAlphabet = pskmod((0:modOrder-1), modOrder);
    otherwise
        disp("Check the status of 'modulationType' above.")
        return % Exit the program early
end

M = 256; % the no. of symbols in a frame
mapType = "interleave"; % type of mapping that is taking place
L = 4*N; % size of IFFT block used on Tx side

Ftx = zeros(N, M); % the transmit frame

% loading the transmit frame with symbols from the signal set
for row = 1:N
    Ftx(row, :) = randsample(modulationAlphabet, M, true);
end

FtxCopy = Ftx; % making a copy before we apply FFT to Ftx
Ftx = fft(Ftx); % taking the FFT before performing the symbol mapping stage

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

% Parameters of targets:
distances = [30 34];
velocities = [60 50];
rcs = [10 15];
alpha = sqrt((c.*rcs)./((4*pi)^3*(distances.^4)*(fC^2)));

numTargets = length(distances);
fD = 2*fC*(velocities)/c;
tau = 2*(distances)/c;

l = 0:M-1;
k = 0:L-1;

dopplerTerm = zeros(numTargets, M);
delayTerm = zeros(numTargets, L);

for row = 1:numTargets
    dopplerTerm(row, :) = exp(1i*2*pi*l*To*fD(row));
    delayTerm(row, :) = exp(-1i*2*pi*k*tau(row)*subcarrierSpacing);
end

Frx = zeros(L, M);
noise = wgn(L, M, -140, 'complex');

for column = 1:M
    for target = 1:numTargets
        Frx(:, column) = Frx(:, column) + alpha(target)*symbolMapOp(:, column)*dopplerTerm(target, column).*delayTerm(target, :).';
    end
end

F = Frx./symbolMapOp;

F(1, :) = 0;
F(L/2 + 1, :) = 0;
F(isnan(abs(F))) = 0;

Nper = 5*L;
Mper = 5*M;

D = 1/10;

Nmax = G*Nper;
Mmax = round(D*Mper);

Cper = F;

Cper = Cper.';
Cper = fft(Cper, Mper);
Cper = Cper.';
Cper = ifft(Cper, Nper);
numCols = size(Cper, 2);
Cper = Cper(1:Nmax, :);
Cper = flip(fftshift(Cper, 2), 1);
Cper = Cper(:, (numCols/2)-Mmax:(numCols/2)+Mmax-1);
Per = 1/(Nmax*(2*Mmax + 1))*(abs(Cper).^2);
maxPer = max(Per(:));
[y, x] = ind2sub(size(Per), find(Per == maxPer));

m_hat = x - (size(Per, 2)/2 + 1);
n_hat = size(Per, 1) - y + 1;

tau_hat = (n_hat - 1)/(Nper*subcarrierSpacing);

distance = (n_hat - 1)*c/(2*subcarrierSpacing*Nper);
velocity = (m_hat - 1)*c/(2*fC*To*Mper);

mIndexes = (1:size(Per, 2)) - (size(Per, 2)/2 + 1);
nIdexes = (1:size(Per, 1));

distancesVec = (nIdexes - 1)*c/(2*subcarrierSpacing*Nper);
velocitiesVec = round((mIndexes - 1)*c/(2*fC*To*Mper));

distances
velocities

distanceLabels = string(distancesVec);
distanceLabels(~(ceil(distancesVec) == floor(distancesVec))) = "";
velocityLabels = string(velocitiesVec);
meanDiff = round(mean(diff(velocitiesVec)));
velocitiesVec2 = min(velocitiesVec):meanDiff:min(velocitiesVec) + meanDiff*(length(velocitiesVec)-1);
velocityLabels2 = string(velocitiesVec2);
velocityLabels(~(mod(velocitiesVec, 5) == 0)) = "";
velocityLabels = replace(velocityLabels, "550", "");
velocityLabels2(~(mod(velocitiesVec2, 10) == 0)) = "";

h = heatmap(velocitiesVec, flip(distancesVec), Per, 'Colormap', jet(300));
s = struct(h);
s.XAxis.TickLabelRotation = 90;
h.YDisplayLabels = flip(distanceLabels);
h.XDisplayLabels = velocityLabels;
set(gca,'Fontname', 'Georgia');
xlabel('\fontname{Georgia}\bf\itv\rm\bf_{rel.} (m/s)');
ylabel('\fontname{Georgia}\bfDistance (m)');