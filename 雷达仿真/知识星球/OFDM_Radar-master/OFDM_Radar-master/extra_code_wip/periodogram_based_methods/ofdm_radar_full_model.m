%% OFDM Radar: SCRIPT NEEDS UPDATING
N = 256; % Size of the IFFT block available at Tx
M = 64; % Order of QAM modulation being used at the Tx...
K = log2(M); % The number of bits each symbol will represent...
codeRate = 0.5; % The code-rate being used by the convolutional encoder at the Tx...
lenData = (N-2)*(1/2)*(K)*(codeRate); % The amount (or length) of the data we should generate for one OFDM symbol...
frameSize = 5; % The number of OFDM symbols per frame...
data = randi([0 1], frameSize*lenData, 1);

txTrellis = poly2trellis(7, [171 133]);
traceBackLength = 34;
encodedData = convenc(data, txTrellis);

% Modulation stage
modulatedData = qammod(encodedData, M, "InputType", "bit", "UnitAveragePower", true);

channelConditions = "Perfect";

switch(channelConditions)
    case "Perfect"
        h = [1; 0; 0]; % Channel coefficients...
    case "Decent"
        h = [0.8; 0.2; 0.4];
    otherwise
        disp("Check status of 'channelConditions' above.")
        return % Exit the program early
end

% The length of the cyclic prefix required is related to the 
% length of the channel...
cpLen = 3*length(h) + 1;

% Transmit frame
Ftx = zeros(N+cpLen, frameSize); % Initially it's empty...

% Tx symbols matrix: this is essentially the transmit frame before
% it's passed to the IFFT block, and before the CP is added...
TxSymbols = zeros(N, frameSize);

n = 1; % A count variable
l = length(modulatedData)/frameSize;

% Loading the transmit frame (Ftx):
for i = 1:l:length(modulatedData)
    ifftInput = [0; modulatedData(i:i+l-1); 0; conj(modulatedData(i+l-1:-1:i))];
    TxSymbols(:, n) = ifftInput; % i.e., the complex symbols
    ifftOutput = ifft(ifftInput); % Passing the symbols into the IFFT block
    Ftx(:, n) = [ifftOutput(end-cpLen+1:end); ifftOutput]; % We are adding the CP in this line...
    n = n + 1;
end

Frx = zeros(N+cpLen+2*(length(h)-1), frameSize); % A matrix to represent our received frame...

SNR = 50; % The SNR of the channel

% Now we'll transmit our frame through the channel:
for column = 1:frameSize
    % Two convolutions: the first is for when the frame
    % is travelling towards the target; the second is for
    % when the reflected wave is passing back through the
    % channel towards the transmitter.
    
    % Adding White Gaussian Noise...
    y = awgn(conv(Ftx(:, column), h), SNR, 'measured');
    % The reflected EM wave passes through the same channel defined by
    % 'h'...
    y = awgn(conv(y, h), SNR, 'measured');
    Frx(:, column) = y; % Filling in each column of our received frame...
end

% Index vectors: l is for the columns, k is for the rows
l = 0:frameSize-1;
k = 0:(N-1) + 2*(length(h) - 1) + cpLen;

% OFDM system parameters:
To = 3.2e-6; % The OFDM symbol period
fC = 5.5e9; % The frequency of the centre subcarrier
subcarrierSpacing = 312.5e3; % The subcarrier spacing

c = 3e8; % The speed of light

% Target parameters:
Vrel = 10; % The relative velocity of our target to the transmitter
d = 10; % The distance between the target and the transmitter
targetRCS = 10; % The radar cross section of the target

% Working out the approximate Doppler shift caused when the EM wave impinges on the
% target:
fD = 2*fC*(Vrel/c);

% And the time delay:
timeDelay = 2*d/c;

% This term occurs in the expression for the received frame:
dopplerTerm = exp(1i*2*pi.*l*To*fD);

% This term also occurs in the expression for the received frame:
delayTerm = exp(-1i*2*pi.*k*timeDelay*subcarrierSpacing)';

% The attenuation factor, b:
b = sqrt((c*targetRCS)/((4*pi)^3*(d^4)*(fC^2)));

% Modeling the effect of the transmitted frame impinging on the target:
for j = 1:frameSize
    Frx(:, j) = b*Frx(:, j).*delayTerm*dopplerTerm(j);
end

% Remove CP and tail from each OFDM symbol (column in the received frame):
Frx = Frx(cpLen+1:end, :); % Removing the CP
Frx = Frx(1:end-2*(length(h) - 1), :); % Removing the tails of each OFDM symbol

% Rx side FFT block:
Frx = fft(Frx); % Taking the FFT of each column of the frame received...

% Idea number 1: ignore the rows with 0 + 0*j in the transmit frame
F = Frx./TxSymbols; % As suggested in Braun Martin's paper...
F(1, :) = Frx(1, :); % Replacing the first row of F, which was full of "Inf" values
F((N/2)+1, :) = Frx((N/2)+1, :); % Doing the same as the line above...

% Periodogram calculation:
% (1): Taking the FFT of each row
F = F.'; % Transposing F, as the fft() function, when used on a matrix, takes the
% FFT of each column... so by transposing the matrix, we turn the rows into
% columns...
F = fft(F, 4*frameSize); % Set size here

F = F.'; % Turning the columns back into rows
F = ifft(F, 4*N); % IFFT of each column

% Now we get the square of the magnitude of each complex value in the
% matrix F:
P = (1/(N*frameSize))*abs(F).^2; % Assigning to P, in order to retain F...

[maxVal, index] = max(P(:));
[iRow, iCol] = ind2sub(size(P), index);

distance = (iRow*c)/(2*subcarrierSpacing*N)
velocity = (iCol*c)/(2*fC*To*frameSize)