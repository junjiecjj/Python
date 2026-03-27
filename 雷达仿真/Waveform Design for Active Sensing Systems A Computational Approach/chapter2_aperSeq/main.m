%% ISL metric (Golomb sequence, m sequence, random-phase sequences)

rng(42); 
addpath('../commons');

M = [5 6 7 8 9 10 11 12 13];
N = 2.^M - 1;
K = length(M);

MFCANGolomb = zeros(K,1);
MFGolomb = zeros(K,1);
MFMseq = zeros(K,1);
MFrand = zeros(K,1);

for k = 1:K
    m = M(k); n = N(k);
    disp(['n = ' num2str(n)]);
    [r ISL] = aperplotsiso(golomb(n));
    MFGolomb(k) = n^2 / ISL;
    [r ISL] = aperplotsiso(cansiso(n,golomb(n)));
    MFCANGolomb(k) = n^2 / ISL;
    [r ISL] = aperplotsiso(mseq(m));
    MFMseq(k) = n^2 / ISL;
    [r ISL] = aperplotsiso(exp(1i*2*pi*rand(n,1)));
    MFrand(k) = n^2 / ISL;
end

figure;
loglog(N, MFCANGolomb, 'ko-', N, MFGolomb, 'bv-', N, MFMseq, 'rd-', N, MFrand, 'ms-');
xlabel('N');
ylabel('Merit Factor');
legend('CAN(G)', 'Golomb', 'm-seq', 'random-phase seq', ...
    'Location', 'NorthWest');
myboldify;
myresize('ISL_all');
% save ISL.mat;

N = 127; M = 7;
r1 = aperplotsiso(cansiso(N,golomb(N)));
r2 = aperplotsiso(mseq(M));
r3 = aperplotsiso(exp(1i*2*pi*rand(N,1)));

figure;
plot(-(N-1):(N-1), 20*log10(abs(r2)/N), 'r--'); hold on;
plot(-(N-1):(N-1), 20*log10(abs(r1)/N), 'k-'); hold off;
xlabel('k'); ylabel('|r(k)|/N (dB)');
legend('m-seq', 'CAN(G)');
axis([-(N-1) N-1 -80 0]);
myboldify;
myresize('CAN_mseq');

figure;
plot(-(N-1):(N-1), 20*log10(abs(r3)/N), 'm--'); hold on;
plot(-(N-1):(N-1), 20*log10(abs(r1)/N), 'k-'); hold off;
xlabel('k'); ylabel('|r(k)|/N (dB)');
legend('random-phase seq', 'CAN(G)');
axis([-(N-1) N-1 -80 0]);
myboldify;
myresize('CAN_randomphase');


aperplotsiso(golomb(100), 'Golomb, N=100', [-80 0]);
myresize('Golomb1');
aperplotsiso(cansiso(100, golomb(100)), 'CAN(G), N=100', [-80 0]);
myresize('CANGolomb1');
aperplotsiso(golomb(1000), 'Golomb, N=1000', [-80 0]);
myresize('Golomb2');
aperplotsiso(cansiso(1000, golomb(1000)), 'CAN(G), N=1000', [-80 0]);
myresize('CANGolomb2');

%% WISL metric 2

N = 100;
weights = zeros(N-1,1); % corresponding to r(1),...,r(N-1)
weights(1:79) = 1; weights(26:69) = 0;
% Golomb
r = aperplotsiso(golomb(N), '');
MMFGolomb = mmf_weight(r(N:end), weights);
% CAN
r = aperplotsiso(cansiso(N,golomb(N)), '');
MMFCAN = mmf_weight(r(N:end), weights);
% WeCAN
xWeCAN = wecansiso(N,[0; weights]);
r = aperplotsiso(xWeCAN, '');
myresize('WeCAN1');
MMFWeCAN = mmf_weight(r(N:end), weights);

disp(['MMF of Golomb: ' num2str(MMFGolomb)]);
disp(['MMF of CAN:    ' num2str(MMFCAN)]);
disp(['MMF of WeCAN:  ' num2str(MMFWeCAN)]);
% save WISL1.mat;

%% FIR channel data preparation (requires 1 day...)

P = 40; % number of channel taps
num = 9;
N_set = 100:50:(50*(num+1));
xWeCAN = zeros(N_set(end),num);
for k = 1:num
    N = N_set(k);
    xWeCAN(1:N,k) = wecansiso(N, [ones(P,1); zeros(N-P,1)]);   
end
% CIR
h = 0.5 * (exp(-0.9*(0:(P-1))') + 1i * exp(-0.9*(0:(P-1))')) + ...
     0.3 * (randn(P,1) + 1i * randn(P,1)) .* exp(-0.1*(0:(P-1))');
save FIR.mat h xWeCAN P num N_set;

figure;
stem(0:(P-1),abs(h));
V = axis;
axis([-1 P V(3) V(4)]);
xlabel('p'); ylabel('       |h_p|');
myboldify;
myresize('CIR');

aperplotsiso(xWeCAN(1:100,1), '');
myresize('WeCAN2');

%% FIR channel estimation 1

load FIR.mat;

noisePower = 0.0001;
MCtrial = 500;
errorGolomb = zeros(num,1); errorWCAN = zeros(num,1);
for k = 1:num
    N = N_set(k);
    XGolomb = zeros(N+P-1,P); XWCAN = zeros(N+P-1,P);
    xGolomb = golomb(N); xWCAN = xWeCAN(1:N,k);
    for p = 1:P
        XGolomb(p:(p+N-1),p) = xGolomb;
        XWCAN(p:(p+N-1),p) = xWCAN;
    end
    errorGolomb_tmp = zeros(MCtrial,1); 
    errorWCAN_tmp = zeros(MCtrial,1);
    for i = 1:MCtrial
        noise = sqrt(noisePower) * (randn(N+P-1,1) + 1i*randn(N+P-1,1))/2;
        yGolomb = XGolomb * h + noise;
        yWCAN = XWCAN * h + noise;
        hGolomb = zeros(P,1); hWCAN = zeros(P,1);
        for p = 1:P
            hGolomb(p) = XGolomb(:,p)' * yGolomb / N;
            hWCAN(p) = XWCAN(:,p)' * yWCAN / N;
        end
        errorGolomb_tmp(i) = (norm(h - hGolomb))^2;
        errorWCAN_tmp(i) = (norm(h - hWCAN))^2;
    end
    errorGolomb(k) = mean(errorGolomb_tmp);
    errorWCAN(k) = mean(errorWCAN_tmp);
end
figure;
semilogy(N_set, errorGolomb, 'bd--', N_set, errorWCAN, 'ms-');
xlabel('N');
ylabel('MSE of Channel Estimate');
legend('Golomb', 'WeCAN', 'Location', 'SouthWest');
axis([N_set(1) N_set(end) 1e-7 1e-1]);
grid on;
myboldify;
myresize('MSE_N');

%% FIR channel estimation 2

load FIR.mat;
N = 200; index = floor(N/50)-1;
NoisePower = [(1:10)*1e-6 (2:10)*1e-5 (2:10)*1e-4 (2:10)*1e-3 (2:10)*0.01 (2:10)*0.1]; 
num = length(NoisePower);
errorGolomb = zeros(num,1); errorWCAN = zeros(num,1);
xGolomb = golomb(N);
xWCAN = xWeCAN(1:N,index);
XGolomb = zeros(N+P-1,P); XWCAN = zeros(N+P-1,P);
for p = 1:P
    XGolomb(p:(p+N-1),p) = xGolomb;
    XWCAN(p:(p+N-1),p) = xWCAN;
end
MCtrial = 500;
for k = 1:num
    errorGolomb_tmp = zeros(MCtrial,1);
    errorWCAN_tmp = zeros(MCtrial,1);
    for i = 1:MCtrial
        noise = sqrt(NoisePower(k)) * (randn(N+P-1,1) + 1i*randn(N+P-1,1))/2;
        yGolomb = XGolomb * h + noise;
        yWCAN = XWCAN * h + noise;
        hGolomb = zeros(P,1); hWCAN = zeros(P,1);
        for p = 1:P
            hGolomb(p) = XGolomb(:,p)' * yGolomb / N;
            hWCAN(p) = XWCAN(:,p)' * yWCAN / N;
        end
        errorGolomb_tmp(i) = (norm(h - hGolomb))^2;
        errorWCAN_tmp(i) = (norm(h - hWCAN))^2;
    end
    errorGolomb(k) = mean(errorGolomb_tmp);
    errorWCAN(k) = mean(errorWCAN_tmp);
end
figure;
loglog(NoisePower, errorGolomb, 'bd--', NoisePower, errorWCAN, 'ms-');
xlabel('Noise Power');
ylabel('MSE of Channel Estimate');
legend('Golomb', 'WeCAN', 'Location', 'SouthEast');
axis([NoisePower(1) NoisePower(end) 1e-7 1e-1]);
grid on;
myboldify;
myresize('MSE_noisepower');

%% Quantization: ISL metric (Golomb sequence, m sequence, random-phase sequences)
M = [5 6 7 8 9 10 11 12 13];
N = 2.^M - 1;
K = length(M);

MFCANGolomb = zeros(K,1);
MFGolomb = zeros(K,1);
MFMseq = zeros(K,1);
MFrand = zeros(K,1);

q = 5; % quantization level
for k = 1:K
    m = M(k); n = N(k);
    disp(['n = ' num2str(n)]);
    [r ISL] = aperplotsiso(golomb(n));
    MFGolomb(k) = n^2 / ISL;
    
    x = cansiso(n,golomb(n));
    x_phase = angle(x);
    index = (x_phase < 0);
    x_phase(index) = x_phase(index) + 2 * pi;
    x = exp(1i * floor(x_phase/(2*pi/(2^q))) * 2*pi/(2^q));
    [r ISL] = aperplotsiso(x);
    
    MFCANGolomb(k) = n^2 / ISL;
    [r ISL] = aperplotsiso(mseq(m));
    MFMseq(k) = n^2 / ISL;
    [r ISL] = aperplotsiso(exp(1i*2*pi*rand(n,1)));
    MFrand(k) = n^2 / ISL;
end

figure;
loglog(N, MFCANGolomb, 'ko-', N, MFGolomb, 'bv-', N, MFMseq, 'rd-', N, MFrand, 'ms-');
xlabel('N');
ylabel('Merit Factor');
legend('Quantized CAN(G)', 'Golomb', 'm-seq', 'random-phase seq', ...
    'Location', 'NorthWest');
myboldify;
myresize('ISL_all_quantized');
% save ISL_quantized.mat;

%% Quantization: WISL metric
load WISL1.mat;
q = 5; % quantization level
x_phase = angle(xWeCAN);
index = (x_phase < 0);
x_phase(index) = x_phase(index) + 2 * pi;
x = exp(1i * floor(x_phase/(2*pi/(2^q))) * 2*pi/(2^q));
aperplotsiso(x,'');
myresize('WeCAN1_quantized');