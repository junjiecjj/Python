%% SCAN
N = 100; % length of the designed sequence
passBand = [0 0.197; 0.303 1]; % each row represents an allowed band
lambda = 0.7; % the weight controls the Stopband/Correlation penalties
rho = 1; % PAR

NTilde = 1000; % length after zero-padding
stopIndex = ones(NTilde, 1); % 1 indicates a frequency that is forbidden
passBandNo = size(passBand, 1);
fGrid = 1/NTilde;
for m = 1:passBandNo
    stopIndex(floor(passBand(m,1)/fGrid)+1 : floor(passBand(m,2)/fGrid)) = 0;
end

x = cansisostoppar(rho, N, NTilde, stopIndex, lambda); % N-by-1

% calculate the spectrum
passBand = [0 0.2; 0.3 1];
K = 1000; % K-point FFT
y = fft(x, K);
fGridShow = 1/K;
passBandShow = zeros(size(passBand));
for m = 1:passBandNo
    passBandShow(m,:) = [floor(passBand(m,1)/fGridShow)+1 ...
        floor(passBand(m,2)/fGridShow)];
end
passBandEnergy = 0;
passBandPoints = 0;
for m = 1:passBandNo
    passBandEnergy = passBandEnergy + sum(abs( ...
        y(passBandShow(m,1):passBandShow(m,2))).^2);
    passBandPoints = passBandPoints + (passBandShow(m,2) - passBandShow(m,1));
end
energyScale = passBandEnergy/passBandPoints;
specDensity = abs(y).^2 / energyScale; % normalize the passband power to 1

% show the spectrum
figure;
plot((0:(K-1))/K, 10*log10(specDensity)); hold on;
axis([0 1 -20 5]);
V = axis;
for m = 1:passBandNo
    plot([passBandShow(m,1)/K passBandShow(m,1)/K], [V(3) V(4)], 'r--'); hold on;
    plot([passBandShow(m,2)/K passBandShow(m,2)/K], [V(3) V(4)], 'r--'); hold on;
end
xlabel('Frequency (Hz)');
ylabel('Spectral Density (dB)');
myboldify;
myresize('spectrum1');

% compare stopband and passband power
stopIndexShow = ones(K, 1);
for m = 1:passBandNo
    stopIndexShow(passBandShow(m,1):passBandShow(m,2)) = 0;
end
PSP = max(specDensity(logical(stopIndexShow))); % peak stopband power
disp('Passband Power = 0 dB');
disp(['Peak Stopband Power = ' num2str(10*log10(PSP)) ' dB']);

% show the auto-correlations
[r isl psl] = aperplotsiso(x, '');
myresize('correlation1');
disp(['PSL = ' num2str(20*log10(psl/N))]);

% save scan1.mat;

%% SCAN vs. lambda/rho
N = 100; % length of the designed sequence
passBand = [0 0.19; 0.31 1]; % each row represents an allowed band
numPar = 20;
lambda = linspace(0.1,1,numPar); % the weight
rho = 1; % PAR

NTilde = 1000; % length after zero-padding
stopIndex = ones(NTilde, 1); % 1 indicates a frequency that is forbidden
passBandNo = size(passBand, 1);
fGrid = 1/NTilde;
for m = 1:passBandNo
    stopIndex(floor(passBand(m,1)/fGrid)+1 : floor(passBand(m,2)/fGrid)) = 0;
end

PSL = zeros(numPar, 1);
PSP = zeros(numPar, 1);
X = zeros(numPar, N);

passBand = [0 0.2; 0.3 1];
K = 1000; % K-point FFT
fGridShow = 1/K;
passBandShow = zeros(size(passBand));
for m = 1:passBandNo
    passBandShow(m,:) = [floor(passBand(m,1)/fGridShow)+1 ...
        floor(passBand(m,2)/fGridShow)];
end

for k = 1:numPar
    disp(['k = ' num2str(k)]); drawnow;
    for trial = 1:100
        x = cansisostoppar(rho, N, NTilde, stopIndex, lambda(k)); % N-by-1        
        % calculate the spectrum        
        y = fft(x, K);        
        passBandEnergy = 0;
        passBandPoints = 0;
        for m = 1:passBandNo
            passBandEnergy = passBandEnergy + sum(abs( ...
                y(passBandShow(m,1):passBandShow(m,2))).^2);
            passBandPoints = passBandPoints + (passBandShow(m,2) - passBandShow(m,1));
        end
        energyScale = passBandEnergy/passBandPoints;
        specDensity = abs(y).^2 / energyScale; % normalize the passband power to 1
        
        stopIndexShow = ones(K, 1);
        for m = 1:passBandNo
            stopIndexShow(passBandShow(m,1):passBandShow(m,2)) = 0;
        end
        psp = max(specDensity(logical(stopIndexShow))); % peak stopband power
        [r isl psl] = aperplotsiso(x);
        
        psp = 10*log10(psp);
        psl = 20*log10(psl/N);
        
        if (psp < PSP(k)) && (psl < PSL(k))
            PSP(k) = psp;
            PSL(k) = psl;
            X(k,:) = x.';
        end
    end
end

figure;
plot(lambda, PSP, 'r-', lambda, PSL, 'b--');
legend('P_{stop}', 'P_{corr}');
xlabel('\lambda');
axis([lambda(1) lambda(end) -50 0]);
myboldify;
myresize('rho1');

% save scan_lambda.mat;

%% SCAN with more stopbands
N = 10000; % length of the designed sequence
passBand = [0.111 0.129; 0.191 0.249; 0.361 0.399; 0.651 0.799; 0.871 0.939];
lambda = 0.9; % the weight controls the Stopband/Correlation penalties
rho = 1; % PAR

NTilde = 10000; % length after zero-padding
stopIndex = ones(NTilde, 1); % 1 indicates a frequency that is forbidden
passBandNo = size(passBand, 1);
fGrid = 1/NTilde;
for m = 1:passBandNo
    stopIndex(floor(passBand(m,1)/fGrid)+1 : floor(passBand(m,2)/fGrid)) = 0;
end 

x = cansisostoppar(rho, N, NTilde, stopIndex, lambda); % N-by-1

% calculate the spectrum
passBand = [0.11 0.13; 0.19 0.25; 0.36 0.40; 0.65 0.8; 0.87 0.94];
K = 10000; % K-point FFT
y = fft(x, K);
fGridShow = 1/K;
passBandShow = zeros(size(passBand));
for m = 1:passBandNo
    passBandShow(m,:) = [floor(passBand(m,1)/fGridShow)+1 ...
        floor(passBand(m,2)/fGridShow)];
end
passBandEnergy = 0;
passBandPoints = 0;
for m = 1:passBandNo
    passBandEnergy = passBandEnergy + sum(abs( ...
        y(passBandShow(m,1):passBandShow(m,2))).^2);
    passBandPoints = passBandPoints + (passBandShow(m,2) - passBandShow(m,1));
end
energyScale = passBandEnergy/passBandPoints;
specDensity = abs(y).^2 / energyScale; % normalize the passband power to 1

% show the spectrum
figure;
plot((0:(K-1))/K, 10*log10(specDensity)); hold on;
V = axis;
for m = 1:passBandNo
    plot([passBandShow(m,1)/K passBandShow(m,1)/K], [V(3) V(4)], 'r--'); hold on;
    plot([passBandShow(m,2)/K passBandShow(m,2)/K], [V(3) V(4)], 'r--'); hold on;
end
xlabel('Frequency (Hz)');
ylabel('Spectral Density (dB)');
myboldify;
myresize('spectrum2');

% compare stopband and passband power
stopIndexShow = ones(K, 1);
for m = 1:passBandNo
    stopIndexShow(passBandShow(m,1):passBandShow(m,2)) = 0;
end
psp = max(specDensity(logical(stopIndexShow))); % peak stopband power
disp('Passband Power = 0 dB');
disp(['Peak Stopband Power = ' num2str(10*log10(psp)) ' dB']);

% show the auto-correlations
[r isl psl] = aperplotsiso(x, '');
myresize('correlation2');
disp(['PSL = ' num2str(20*log10(psl/N))]);

% save scan2.mat;

%% WeSCAN
N = 100; % length of the designed sequence
passBand = [0 0.19; 0.31 1]; % each row represents an allowed band
lambda = 0.7; % the weight controls the Stopband/Correlation penalties
rho = 1; % PAR

NTilde = 1000; % length after zero-padding
stopIndex = ones(NTilde, 1); % 1 indicates a frequency that is forbidden
passBandNo = size(passBand, 1);
fGrid = 1/NTilde;
for m = 1:passBandNo
    stopIndex(floor(passBand(m,1)/fGrid)+1 : floor(passBand(m,2)/fGrid)) = 0;
end

gamma = ones(N,1);
gamma(2:3) = 0;
x = wecansisostoppar(rho, N, gamma, NTilde, stopIndex, lambda); % N-by-1

% calculate the spectrum
passBand = [0 0.2; 0.3 1]; 
K = 1000; % K-point FFT
y = fft(x, K);
fGridShow = 1/K;
passBandShow = zeros(size(passBand));
for m = 1:passBandNo
    passBandShow(m,:) = [floor(passBand(m,1)/fGridShow)+1 ...
        floor(passBand(m,2)/fGridShow)];
end
passBandEnergy = 0;
passBandPoints = 0;
for m = 1:passBandNo
    passBandEnergy = passBandEnergy + sum(abs( ...
        y(passBandShow(m,1):passBandShow(m,2))).^2);
    passBandPoints = passBandPoints + (passBandShow(m,2) - passBandShow(m,1));
end
energyScale = passBandEnergy/passBandPoints;
specDensity = abs(y).^2 / energyScale; % normalize the passband power to 1

% show the spectrum
figure;
plot((0:(K-1))/K, 10*log10(specDensity)); hold on;
V = axis;
for m = 1:passBandNo
    plot([passBandShow(m,1)/K passBandShow(m,1)/K], [V(3) V(4)], 'r--'); hold on;
    plot([passBandShow(m,2)/K passBandShow(m,2)/K], [V(3) V(4)], 'r--'); hold on;
end
ylim([-60 10]);
xlabel('Frequency (Hz)');
ylabel('Spectral Density (dB)');
myboldify;
myresize('spectrum3');

% compare stopband and passband power
stopIndexShow = ones(K, 1);
for m = 1:passBandNo
    stopIndexShow(passBandShow(m,1):passBandShow(m,2)) = 0;
end
psp = max(specDensity(logical(stopIndexShow))); % peak stopband power
disp('Passband Power = 0 dB');
disp(['Peak Stopband Power = ' num2str(10*log10(psp)) ' dB']);

% show the auto-correlations
r = aperplotsiso(x, '');
myresize('correlation3');
psl = max(abs(r(1:N-3)));
disp(['PSL = ' num2str(20*log10(psl/N))]);

% save wescan.mat;

%% SCAN when rho>1
N = 100; % length of the designed sequence
passBand = [0 0.197; 0.303 1]; % each row represents an allowed band
lambda = 0.7; % the weight controls the Stopband/Correlation penalties
rho = 2; % PAR

NTilde = 1000; % length after zero-padding
stopIndex = ones(NTilde, 1); % 1 indicates a frequency that is forbidden
passBandNo = size(passBand, 1);
fGrid = 1/NTilde;
for m = 1:passBandNo
    stopIndex(floor(passBand(m,1)/fGrid)+1 : floor(passBand(m,2)/fGrid)) = 0;
end

x = cansisostoppar(rho, N, NTilde, stopIndex, lambda); % N-by-1

% calculate the spectrum
passBand = [0 0.2; 0.3 1];
K = 1000; % K-point FFT
y = fft(x, K);
fGridShow = 1/K;
passBandShow = zeros(size(passBand));
for m = 1:passBandNo
    passBandShow(m,:) = [floor(passBand(m,1)/fGridShow)+1 ...
        floor(passBand(m,2)/fGridShow)];
end
passBandEnergy = 0;
passBandPoints = 0;
for m = 1:passBandNo
    passBandEnergy = passBandEnergy + sum(abs( ...
        y(passBandShow(m,1):passBandShow(m,2))).^2);
    passBandPoints = passBandPoints + (passBandShow(m,2) - passBandShow(m,1));
end
energyScale = passBandEnergy/passBandPoints;
specDensity = abs(y).^2 / energyScale; % normalize the passband power to 1

% show the spectrum
figure;
plot((0:(K-1))/K, 10*log10(specDensity)); hold on;
axis([0 1 -20 5]);
V = axis;
for m = 1:passBandNo
    plot([passBandShow(m,1)/K passBandShow(m,1)/K], [V(3) V(4)], 'r--'); hold on;
    plot([passBandShow(m,2)/K passBandShow(m,2)/K], [V(3) V(4)], 'r--'); hold on;
end
xlabel('Frequency (Hz)');
ylabel('Spectral Density (dB)');
myboldify;
myresize('spectrum5');

% compare stopband and passband power
stopIndexShow = ones(K, 1);
for m = 1:passBandNo
    stopIndexShow(passBandShow(m,1):passBandShow(m,2)) = 0;
end
PSP = max(specDensity(logical(stopIndexShow))); % peak stopband power
disp('Passband Power = 0 dB');
disp(['Peak Stopband Power = ' num2str(10*log10(PSP)) ' dB']);

% show the auto-correlations
[r isl psl] = aperplotsiso(x, '');
myresize('correlation5');
disp(['PSL = ' num2str(20*log10(psl/N))]);

%save scan_rho.mat;

%% SCAN using a continuous stopband formulation
N = 100;
lambda = 0.7;
f1 = 0.2; f2 = 0.3;
R = zeros(N,N);
for m = 1:N
    for n = 1:N
        if m == n
            R(m,n) = f2 - f1;
        else
            R(m,n) = (exp(1i*2*pi*(m-n)*f2) - exp(1i*2*pi*(m-n)*f1))/(1i*2*pi*(m-n));
        end
    end
end
[U D] = svd(R);
index = (abs(diag(D)) < 1e-6);
S = U(:,index);

x = exp(1i * 2*pi * rand(N,1));
xPrev = zeros(N,1);

iterDiff = norm(x - xPrev);
while (iterDiff > 1e-3)
    xPrev = x;
    disp(['diff = ' num2str(iterDiff)]);
    % w.r.t v
    z = [x; zeros(N,1)];
    f = 1/sqrt(2*N) * fft(z);
    v = 1/sqrt(2) * exp(1i * angle(f)); % 2N-by-1
    % w.r.t alpha
    alpha = S' * x;
    % w.r.t x
    vFFT = sqrt(2*N) * ifft(v); % 2N-by-1
    h = lambda * S * alpha + (1 - lambda) * vFFT(1:N);
    x = exp(1i * angle(h));
   
    iterDiff = norm(x - xPrev);
end

% calculate the spectrum
passBand = [0 0.2; 0.3 1];
passBandNo = size(passBand, 1);
K = 1000; % K-point FFT
y = fft(x, K);
fGridShow = 1/K;
passBandShow = zeros(size(passBand));
for m = 1:passBandNo
    passBandShow(m,:) = [floor(passBand(m,1)/fGridShow)+1 ...
        floor(passBand(m,2)/fGridShow)];
end
passBandEnergy = 0;
passBandPoints = 0;
for m = 1:passBandNo
    passBandEnergy = passBandEnergy + sum(abs( ...
        y(passBandShow(m,1):passBandShow(m,2))).^2);
    passBandPoints = passBandPoints + (passBandShow(m,2) - passBandShow(m,1));
end
energyScale = passBandEnergy/passBandPoints;
specDensity = abs(y).^2 / energyScale; % normalize the passband power to 1

% show the spectrum
figure;
plot((0:(K-1))/K, 10*log10(specDensity)); hold on;
V = axis;
V(3) = -20; V(4) = 5;
for m = 1:passBandNo
    plot([passBandShow(m,1)/K passBandShow(m,1)/K], [V(3) V(4)], 'r--'); hold on;
    plot([passBandShow(m,2)/K passBandShow(m,2)/K], [V(3) V(4)], 'r--'); hold on;
end
xlabel('Frequency (Hz)');
ylabel('Spectral Density (dB)');
axis([0 1 -20 5]);
myboldify;
myresize('spectrum4');

% compare stopband and passband power
stopIndexShow = ones(K, 1);
for m = 1:passBandNo
    stopIndexShow(passBandShow(m,1):passBandShow(m,2)) = 0;
end
PSP = max(specDensity(logical(stopIndexShow))); % peak stopband power
disp('Passband Power = 0 dB');
disp(['Peak Stopband Power = ' num2str(10*log10(PSP)) ' dB']);

% show the auto-correlations
[r isl psl] = aperplotsiso(x, '');
myresize('correlation4');
disp(['PSL = ' num2str(20*log10(psl/N))]);

% save scan_continuous.mat;

%% SCAN using a continuous stopband formulation -- two stopbands
% the weights do not matter because as long as the weights are non-zero,
% they do not impact the null space of R
N = 100;
lambda = 0.7;
stopband = [0.2 0.3; 0.6 0.8];
weight = [0.99; 0.01];
R = zeros(N,N);
for m = 1:N
    for n = 1:N
        if m == n
            R(m,n) = weight' * (stopband(:,2) - stopband(:,1));
        else
            R(m,n) = weight' * ((exp(1i*2*pi*(m-n)*stopband(:,2)) ...
                - exp(1i*2*pi*(m-n)*stopband(:,1))) / (1i*2*pi*(m-n)));
        end
    end
end
[U D] = svd(R);
index = (abs(diag(D)) < 1e-6);
S = U(:,index);

x = exp(1i * 2*pi * rand(N,1));
xPrev = zeros(N,1);

iterDiff = norm(x - xPrev);
while (iterDiff > 1e-3)
    xPrev = x;
    disp(['diff = ' num2str(iterDiff)]);
    % w.r.t v
    z = [x; zeros(N,1)];
    f = 1/sqrt(2*N) * fft(z);
    v = 1/sqrt(2) * exp(1i * angle(f)); % 2N-by-1
    % w.r.t alpha
    alpha = S' * x;
    % w.r.t x
    vFFT = sqrt(2*N) * ifft(v); % 2N-by-1
    h = lambda * S * alpha + (1 - lambda) * vFFT(1:N);
    x = exp(1i * angle(h));
   
    iterDiff = norm(x - xPrev);
end

% calculate the spectrum
passBand = [0 0.2; 0.3 0.6; 0.8 1];
passBandNo = size(passBand, 1);
K = 1000; % K-point FFT
y = fft(x, K);
fGridShow = 1/K;
passBandShow = zeros(size(passBand));
for m = 1:passBandNo
    passBandShow(m,:) = [floor(passBand(m,1)/fGridShow)+1 ...
        floor(passBand(m,2)/fGridShow)];
end
passBandEnergy = 0;
passBandPoints = 0;
for m = 1:passBandNo
    passBandEnergy = passBandEnergy + sum(abs( ...
        y(passBandShow(m,1):passBandShow(m,2))).^2);
    passBandPoints = passBandPoints + (passBandShow(m,2) - passBandShow(m,1));
end
energyScale = passBandEnergy/passBandPoints;
specDensity = abs(y).^2 / energyScale; % normalize the passband power to 1

% show the spectrum
figure;
plot((0:(K-1))/K, 10*log10(specDensity)); hold on;
V = axis;
V(3) = -20; V(4) = 5;
for m = 1:passBandNo
    plot([passBandShow(m,1)/K passBandShow(m,1)/K], [V(3) V(4)], 'r--'); hold on;
    plot([passBandShow(m,2)/K passBandShow(m,2)/K], [V(3) V(4)], 'r--'); hold on;
end
xlabel('Frequency (Hz)');
ylabel('Spectral Density (dB)');
axis([0 1 -20 5]);
myboldify;
myresize('spectrum6');

% compare stopband and passband power
stopIndexShow = ones(K, 1);
for m = 1:passBandNo
    stopIndexShow(passBandShow(m,1):passBandShow(m,2)) = 0;
end
PSP = max(specDensity(logical(stopIndexShow))); % peak stopband power
disp('Passband Power = 0 dB');
disp(['Peak Stopband Power = ' num2str(10*log10(PSP)) ' dB']);

% show the auto-correlations
[r isl psl] = aperplotsiso(x, '');
myresize('correlation6');
disp(['PSL = ' num2str(20*log10(psl/N))]);

% save scan_continuous_twobands.mat;

