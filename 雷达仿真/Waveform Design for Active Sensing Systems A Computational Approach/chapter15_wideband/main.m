%% phased array
N = 64;
M = 10;
fc = 1 * 10^9; % 1GHz
B = 200 * 10^6; % 100MHz
K = 181;
theta0 = 120;
Pat = phaseArrayPattern(N, M, fc, B, K, theta0);
% myresize('ex1_2D');
% myresize('ex1_3D');

%% wideband signal synthesis - example 1, one narrow mainlobe
N = 64;
M = 10;
fc = 1 * 10^9; % 1GHz
B = 200 * 10^6;
fs = 200 * 10^6;
f_low = fc - fs/2;
K = 181;
spaGrid = 180/(K-1); % degrees per grid
freGrid = fs/N; % Hz per grid
dePattern = zeros(N,K);

deFreq = [fc-B/2, fc+B/2];
deSpat = [120 120];

for k = 1:size(deFreq,1)
   fband_low = floor((deFreq(k,1) - f_low)/freGrid)+1;
   fband_high = floor((deFreq(k,2) - f_low)/freGrid)+1;
   if fband_high > N
       fband_high = N;
   end
   sband_low = floor(deSpat(k,1)/spaGrid)+1;
   sband_high = floor(deSpat(k,2)/spaGrid)+1;
   if sband_high > K
       sband_high = K;
   end
   dePattern(fband_low:fband_high, sband_low:sband_high) = 1;
end
%dePattern = dePattern * sqrt(M^2 * N);

[X FS] = wican(N,M,dePattern,fc,B,fs,K);
Pat1 = freSpaPattern(FS,fc,B,fs,K, 'ex2free_2D', 'ex2free_3D'); % FS. Pat is N-by-K
Pat2 = freSpaPattern(X,fc,B,fs,K, 'ex2uni_2D', 'ex2uni_3D'); % X. Pat is N-by-K
metric1 = sum(sum((abs(dePattern - Pat1)).^2));
metric2 = sum(sum((abs(dePattern - Pat2)).^2));
disp(['Metric - Energy constraint only: ' num2str(metric1)]);
disp(['Metric - Unit-modulus constraint: ' num2str(metric2)]);
for m = 1:M
    disp(['Energy(m=' num2str(m) '): ' num2str(norm(FS(:,m))^2) ...
        ' PAR(m=' num2str(m) '): '...
        num2str((max(abs(FS(:,m))))^2 / ((norm(FS(:,m)))^2 / N))]);
end
freSpaPatCont(X,fc,B,fs,K,fs*5, [-100*10^6 100*10^6], ...
    'ex2spectrum', 'ex2cont_2D', 'ex2cont_3D'); % Continuous

%% wideband signal synthesis - example 2, two mainlobes at two frequencies
N = 64;
M = 10;
fc = 1 * 10^9; % 1GHz
B = 200 * 10^6;
fs = 200 * 10^6;
f_low = fc - fs/2;
K = 181;
spaGrid = 180/(K-1); % degrees per grid
freGrid = fs/N; % Hz per grid
dePattern = zeros(N,K);

deFreq = [fc-B/2, fc; fc, fc+B/2]; % frequency bands
deSpat = 90 + [30 30; -30 -30]; % spatial degrees

for k = 1:size(deFreq,1)
   fband_low = floor((deFreq(k,1) - f_low)/freGrid)+1;
   fband_high = floor((deFreq(k,2) - f_low)/freGrid)+1;
   if fband_high > N
       fband_high = N;
   end
   sband_low = floor(deSpat(k,1)/spaGrid)+1;
   sband_high = floor(deSpat(k,2)/spaGrid)+1;
   if sband_high > K
       sband_high = K;
   end
   dePattern(fband_low:fband_high, sband_low:sband_high) = 1;
end
dePattern = dePattern * sqrt(M^2 * N);

X = wican(N,M,dePattern,fc,B,fs,K);
Pat1 = freSpaPattern(X,fc,B,fs,K,'ex3uni_2D','ex3uni_3D'); % Pat is N-by-K
X2 = wican(N,M,dePattern,fc,B,fs,K, 2);
Pat2 = freSpaPattern(X2,fc,B,fs,K,'ex3par_2D','ex3par_3D'); % Pat is N-by-K
metric1 = sum(sum((abs(dePattern - Pat1)).^2));
metric2 = sum(sum((abs(dePattern - Pat2)).^2));
disp(['Metric - PAR=1: ' num2str(metric1)]);
disp(['Metric - PAR=2: ' num2str(metric2)]);

%% wideband signal synthesis - example 3, one wide mainlobe
N = 64;
M = 10;
fc = 1 * 10^9; % 1GHz
B = 200 * 10^6;
fs = 200 * 10^6;
f_low = fc - fs/2;
f_high = fc + fs/2;
K = 181;
spaGrid = 180/(K-1); % degrees per grid
freGrid = fs/N; % Hz per grid
dePattern = zeros(N,K);

deFreq = [fc-B/2, fc+B/2];
deSpat = [100 140];

for k = 1:size(deFreq,1)
   fband_low = floor((deFreq(k,1) - f_low)/freGrid)+1;
   fband_high = floor((deFreq(k,2) - f_low)/freGrid)+1;
   if fband_high > N
       fband_high = N;
   end
   sband_low = floor(deSpat(k,1)/spaGrid)+1;
   sband_high = floor(deSpat(k,2)/spaGrid)+1;
   if sband_high > K
       sband_high = K;
   end
   dePattern(fband_low:fband_high, sband_low:sband_high) = 1;
end
dePattern = dePattern * sqrt(M^2 * N);

X = wican(N,M,dePattern,fc,B,fs,K);
Pat1 = freSpaPattern(X,fc,B,fs,K,'ex4wide_2D','ex4wide_3D'); % Pat is N-by-K
X2 = wican(N,M,dePattern,fc,B,fs,K, 2);
Pat2 = freSpaPattern(X2,fc,B,fs,K,'ex4widePAR_2D','ex4widePAR_3D'); % Pat is N-by-K
metric1 = sum(sum((abs(dePattern - Pat1)).^2));
metric2 = sum(sum((abs(dePattern - Pat2)).^2));
disp(['Metric - PAR=1: ' num2str(metric1)]);
disp(['Metric - PAR=2: ' num2str(metric2)]);

%% wideband signal synthesis - example 4, larger frequency band
N = 64;
M = 10;
fc = 1 * 10^9; % 1GHz
B = 350 * 10^6;
fs = 350 * 10^6;
f_low = fc - fs/2;
K = 181;
spaGrid = 180/(K-1); % degrees per grid
freGrid = fs/N; % Hz per grid
dePattern = zeros(N,K);

deFreq = [fc-B/2, fc+B/2];
deSpat = [120 120];

for k = 1:size(deFreq,1)
   fband_low = floor((deFreq(k,1) - f_low)/freGrid)+1;
   fband_high = floor((deFreq(k,2) - f_low)/freGrid)+1;
   if fband_high > N
       fband_high = N;
   end
   sband_low = floor(deSpat(k,1)/spaGrid)+1;
   sband_high = floor(deSpat(k,2)/spaGrid)+1;
   if sband_high > K
       sband_high = K;
   end
   dePattern(fband_low:fband_high, sband_low:sband_high) = 1;
end
dePattern = dePattern * sqrt(M^2 * N);

X = wican(N,M,dePattern,fc,B,fs,K);
Pat1 = freSpaPattern(X,fc,B,fs,K,'ex4band_2D','ex4band_3D'); % Pat is N-by-K
metric1 = sum(sum((abs(dePattern - Pat1)).^2));
disp(['Metric - Unit-modulus: ' num2str(metric1)]);
