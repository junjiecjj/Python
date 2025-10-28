%% Multiple Target Estimation with Periodogram:
N = 52; % The number of subcarriers being used
frameSize = 256; % The frame size
M = 16; % Order of QAM scheme being used
d = (0:M-1)';

% General constants:
c = 3e8; % The speed of light

% OFDM system parameters:
To = 4e-6; % The OFDM symbol period
fC = 5.5e9; % The frequency of the centre subcarrier
subcarrierSpacing = 312.5e3; % The subcarrier spacing

% Parameters of targets:
targetDistances = [30 34];
targetRelVelocities = [60 50];
targetRCSs = [10 15];
bVec = sqrt((c.*targetRCSs)./((4*pi)^3*(targetDistances.^4)*(fC^2))); % The attenuations associated with the various targets...

numTargets = length(targetDistances);

% Doppler shift and time delay values:
dopplerShifts = 2*fC*(targetRelVelocities)/c;
timeDelays = 2*(targetDistances)/c;

% Index vectors: l is for the columns, k is for the rows
l = 0:frameSize-1;
k = 0:(N-1);

dopplerTermMatrix = zeros(numTargets, frameSize);
delayTermMatrix = zeros(numTargets, N);

for row = 1:numTargets
    dopplerTermMatrix(row, :) = exp(1i*2*pi*l*To*dopplerShifts(row)).';
    delayTermMatrix(row, :) = exp(-1i*2*pi*k*timeDelays(row)*subcarrierSpacing).';
end

modulationType = "QAM";

switch(modulationType)
    case "QAM"
        signalSet = qammod(d, M); % Generate the signal set for the relevant QAM scheme (16, 32, 64 etc.)     
    case "QPSK"
        signalSet = pskmod(d, M);
    otherwise
        disp("Check the status of 'modulationType' above.")
        return % Exit the program early
end

Ftx = zeros(N/2, frameSize); % Half of the empty transmit frame

for row = 2:N/2 % We start at row 2, as row 1 is empty for the DC subcarrier...
    Ftx(row, :) = randsample(signalSet, frameSize, true);
end

% The full transmit frame. Here we are imposing Hermitian symmetry, so that
% the output from the IFFT block is real-valued...
Ftx = [Ftx; zeros(1, frameSize); conj(flip(Ftx(2:end, :), 1))];

Frx = zeros(N, frameSize); % The received frame (currently empty)
Z = wgn(N, frameSize, -140, 'complex'); % Complex Gaussian White noise matrix...

% Next, we form the receive frame by modelling the effect of the channel
% and the target on the transmited frame:
% EDIT: can be emulate the summation of many reflections by having a nested
% loop. Although there would be a dopplerTerm and delayTerm vector
% associated with each of the targets. All of these vectors could be stored
% inside a matrix...
for column = 1:frameSize
    for target = 1:numTargets
        Frx(:, column) = Frx(:, column) + bVec(target)*Ftx(:, column)*dopplerTermMatrix(target, column).*delayTermMatrix(target, :).';
    end
end

% The OFDM beacon knows what frame was sent, and hence we can divide the
% received frame by the transmitted one:
F = Frx./Ftx;

% And now we can remove the rows with "Inf" values, which were caused
% by dividing by "0"s in the transmit frame (Ftx)
F(1, :) = 0;
F(N/2 + 1, :) = 0;

% Size of FFT and IFFT calculations to be done on the rows and columns of
% the matrix F; these are also the number of rows and columns of the
% uncropped periodogram. Increasing the number of rows and columns in the
% FFT and IFFT calculations increases the resolution, but it also increases
% the number of computations needing to be done...
Nper = 5*N;
Mper = 5*frameSize;

G = 1/8; % Fraction of OFDM symbol used as a guard interval (TG/T)

% The subcarrier spacing should be at least one order of magnitude
% bigger than the largest occurring Doppler shift - hence a value of 1/10 
% for the quantity D...
D = 1/10;

Nmax = G*Nper; % The row at beyond which we should stop looking for peaks
Mmax = round(D*Mper); % The column beyond which we should stop searching for peaks

% Complex periodogram calculation:
Cper = F; % The complex periodogram is calculated using F...

% Method 1 to calculate the complex periodogram:
Cper = Cper.';
Cper = fft(Cper, Mper); % FFT of each row (length of FFT is Mper)
Cper = Cper.';
Cper = ifft(Cper, Nper); % IFFT of each column...
numCols = size(Cper, 2);
Cper = Cper(1:Nmax, :);
Cper = flip(fftshift(Cper, 2), 1);
Cper = Cper(:, (numCols/2)-Mmax:(numCols/2)+Mmax-1);

% Periodogram calculation:
Per = 1/(Nmax*(2*Mmax + 1))*(abs(Cper).^2);

% Getting the maximum value from the cropped periodogram:
maxPer = max(Per(:));

% The x and y axis are reversed in the periodogram...
[y, x] = ind2sub(size(Per), find(Per == maxPer))

% The position indexes. I'm using 'hat' in the variable name as these are
% estimates...
m_hat = x - (size(Per, 2)/2 + 1);
n_hat = size(Per, 1) - y + 1;

tau_hat = (n_hat - 1)/(Nper*subcarrierSpacing) % The time delay...

distance = (n_hat - 1)*c/(2*subcarrierSpacing*Nper)
velocity = (m_hat - 1)*c/(2*fC*To*Mper)

% The index vectors below can be used to find the distance
% and velocity values, which can be used to label the axes...
mIndexes = (1:size(Per, 2)) - (size(Per, 2)/2 + 1);
nIdexes = (1:size(Per, 1));

% Working out the corresponding distance and velocity vector
% associated with each (n, m) index...
distancesVec = (nIdexes - 1)*c/(2*subcarrierSpacing*Nper);
velocitiesVec = round((mIndexes - 1)*c/(2*fC*To*Mper));

meanDiff1 = round(mean(diff(distancesVec)));
meanDiff2 = round(mean(diff(velocitiesVec)));
distancesVec2 = min(distancesVec):meanDiff1:min(distancesVec) + meanDiff1*(length(distancesVec)-1);
velocitiesVec2 = min(velocitiesVec):meanDiff2:min(velocitiesVec) + meanDiff2*(length(velocitiesVec)-1);
distanceLabels2 = string(distancesVec2);
% Creating label variables to use with the heatmap object below:
distanceLabels = string(distancesVec);
distanceLabels(~(ceil(distancesVec) == floor(distancesVec))) = "";
velocityLabels = string(velocitiesVec);
velocityLabels2 = string(velocitiesVec2);
velocityLabels2(~(mod(velocitiesVec2, 6) == 0)) = "";
velocityLabels(~(mod(velocitiesVec, 5) == 0)) = "";

h = heatmap(velocitiesVec2, flip(distancesVec2), Per, 'Colormap', jet(300));
s = struct(h);
s.XAxis.TickLabelRotation = 90;
h.YDisplayLabels = flip(distanceLabels2);
h.XDisplayLabels = velocityLabels2;
set(gca,'Fontname', 'Georgia');
%title('\fontsize{14}\fontname{Georgia}Periodogram Target Estimation');
xlabel('\fontname{Georgia}\bf\itv\rm\bf_{rel.} (m/s)');
ylabel('\fontname{Georgia}\bfDistance (m)');