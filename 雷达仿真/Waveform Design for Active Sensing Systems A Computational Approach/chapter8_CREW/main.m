%% Unimodularity constraint
% CAN, CAN(IV), CREW(gra) and bound

clear; clc;
PAR = 1;
jammertype = 'spot';
%jammertype = 'barrage';
%jammertype = 'white';

NSet = [25 50 100 200 300]';
num = length(NSet);
mseCAN = zeros(num, 1);
mseCANIV = zeros(num, 1);
mseGRA = zeros(num, 1); sGRA = zeros(NSet(end), num);
msebound = zeros(num, 1);

for k = 1:num
    N = NSet(k); disp(['N = ' num2str(N)]);
    s0 = golomb(N);    
    %disp('  CREW(fre)-PAR');
    [s w mse msebound(k)] = crew(N, 'FRE', jammertype, PAR, s0);    
    disp('  CAN');
    [s w mseCAN(k)] = crew(N, 'CAN', jammertype, PAR, s0);    
    disp('  CAN(IV)');
    [s w mseCANIV(k)] = crew(N, 'IVC', jammertype, PAR, s0);    
    disp('  CREW(gra)');
    [sGRA(1:N,k) w mseGRA(k)] = crew(N, 'GRA', jammertype, PAR, s0);
end

figure;
semilogy(NSet, mseCAN, '--ro', ...
    NSet, mseCANIV, '-ro', ...
    NSet, mseGRA, '-bd', ...
    NSet, msebound, '--kx');
legend('CAN', 'CAN-IV', 'CREW(gra)', 'Lower Bound');
xlabel('N');
ylabel('MSE');
title(['MSE vs. N (' jammertype ' jamming, PAR=1)']);
myboldify;
name = [jammertype '_' num2str(PAR)];
myresize(name);

% save([name '.mat']);

%% plot one CREW(gra) waveform power spectrum
load spot_1.mat;
sGRA = sGRA(1:100,3); % a length-100 CREW(gra) sequence
N = 100;
sigma2 = 0.1; % noise power
sigma2jammer = 100; % jammer power
beta = 1;
rho = 2;
s0 = golomb(N);
jammertype = 'spot';

switch jammertype
    case 'white'
        Gamma = (sigma2 + sigma2jammer) * eye(N);
    case 'spot'
        f1 = 0.2;
        Num = 2*N-1;
        power = zeros(Num,1);
        power(floor(f1*Num)+1) = 1;
        r = ifft(power); % [r(0) r(1) ... r(N-1) r*(N-1) ... r*(1)].'
        r = r/abs(r(1)) * sigma2jammer;
        Gamma = toeplitz(r(1:N), [r(1) r(2*N-1:-1:N+1).']) ...
            + sigma2 * eye(N);
    case 'barrage'
        f1 = 0.2; f2 = 0.3;
        Num = 2*N-1;
        power = zeros(Num,1);
        power(floor(f1*Num)+1 : floor(f2*Num)) = 1;
        r = ifft(power); % [r(0) r(1) ... r(N-1) r*(N-1) ... r*(1)].'
        r = r/abs(r(1)) * sigma2jammer;
        Gamma = toeplitz(r(1:N), [r(1) r(2*N-1:-1:N+1).']) ...
            + sigma2 * eye(N);
    otherwise
        error('Unrecognized jammer type');
end
show_criterion(sGRA, Gamma, beta, 'GRA');
% myresize('spot_1_GRA_N_100_corr');
% myresize('spot_1_GRA_N_100_seq'); % use ylim([-40 0])
% myresize('spot_1_GRA_N_100_rx'); % use ylim([-40 0])


%% PAR constraint
% CAN, CAN(IV), CREW(mat) and bound

clear; clc;
PAR = 2;
%jammertype = 'spot';
jammertype = 'barrage';
%jammertype = 'white';

NSet = [25 50 100 200 300 500 1000]';
%NSet = [25 50 100 200];
num = length(NSet);
mseCAN = zeros(num, 1);
mseCANIV = zeros(num, 1);
mseFRE = zeros(num, 1);
mseMAT = zeros(num, 1);
% mseCYC = zeros(num, 1);
msebound = zeros(num, 1);

for k = 1:num
    N = NSet(k); disp(['N = ' num2str(N)]);
    s0 = golomb(N);    
    
    [s w mseFRE(k) msebound(k)] = crew(N, 'FRE', jammertype, PAR, s0);
    disp(['  CREW(fre) (PAR = ' num2str(par(s)) ')']); 
    
    [s w mseCAN(k)] = crew(N, 'CAN', jammertype, PAR, s0); 
    disp(['  CAN (PAR = ' num2str(par(s)) ')']);    
    
    [s w mseCANIV(k)] = crew(N, 'IVC', jammertype, PAR, s0);
    disp(['  CAN-IV (PAR = ' num2str(par(s)) ')']);    
    
    [s w mseMAT(k)] = crew(N, 'MAT', jammertype, PAR, s0);
    disp(['  CREW(mat) (PAR = ' num2str(par(s)) ')']);
    
%     [s w mseCYC(k)] = crew(N, 'CYC', jammertype, PAR, s0);
%     disp(['  CREW(cyc) (PAR = ' num2str(par(s)) ')']);
end

% figure;
% semilogy(NSet, mseCAN, '--ro', ...
%     NSet, mseCANIV, '-ro', ...
%     NSet, mseFRE, '-bv', ...
%     NSet, mseMAT, '-b^', ...
%     NSet, msebound, '--kx');
% legend('CAN', 'CAN-IV', 'CREW(fre)', ...
%     'CREW(mat)', 'CREW(fre)-zero', 'Lower Bound');

% for spot_2.mat
% semilogy(NSet, mseCAN, '--ro', ...
%     NSet, mseCANIV, '-ro', ...
%     NSet, mseMAT, '-b^', ...
%     NSet, msebound, '--kx');
% legend('CAN', 'CAN-IV', 'CREW(mat)', 'Lower Bound');

% for barrage_2.mat
semilogy(NSet, mseCAN, '--ro', ...
    NSet, mseCANIV, '-ro', ...
    NSet, mseFRE, '-bv', ...
    NSet, msebound, '--kx');
legend('CAN', 'CAN-IV', 'CREW(fre)', 'Lower Bound');

xlabel('N');
ylabel('MSE');
title(['MSE vs. N (' jammertype ' jamming, PAR<=' num2str(PAR) ')']);
myboldify;

name = [jammertype '_' num2str(PAR)];
myresize(name);
% save([name '.mat']);

%% plot one CREW(fre) waveform power spectrum

N = 300;
sigma2 = 0.1; % noise power
sigma2jammer = 100; % jammer power
beta = 1;
rho = 2;
s0 = golomb(N);
jammertype = 'barrage';

switch jammertype
    case 'white'
        Gamma = (sigma2 + sigma2jammer) * eye(N);
    case 'spot'
        f1 = 0.2;
        Num = 2*N-1;
        power = zeros(Num,1);
        power(floor(f1*Num)+1) = 1;
        r = ifft(power); % [r(0) r(1) ... r(N-1) r*(N-1) ... r*(1)].'
        r = r/abs(r(1)) * sigma2jammer;
        Gamma = toeplitz(r(1:N), [r(1) r(2*N-1:-1:N+1).']) ...
            + sigma2 * eye(N);
    case 'barrage'
        f1 = 0.2; f2 = 0.3;
        Num = 2*N-1;
        power = zeros(Num,1);
        power(floor(f1*Num)+1 : floor(f2*Num)) = 1;
        r = ifft(power); % [r(0) r(1) ... r(N-1) r*(N-1) ... r*(1)].'
        r = r/abs(r(1)) * sigma2jammer;
        Gamma = toeplitz(r(1:N), [r(1) r(2*N-1:-1:N+1).']) ...
            + sigma2 * eye(N);
    otherwise
        error('Unrecognized jammer type');
end

[sFRE wFRE mseFRE] = cyc_freq(N, Gamma, beta, rho, s0);
show_criterion(sFRE, Gamma, beta, 'FRE');
% save barrage_fre_300.mat;

% myresize('barrage_2_FRE_N_300_corr');
% myresize('barrage_2_FRE_N_300_seq'); % use ylim([-40 0])
% myresize('barrage_2_FRE_N_300_rx'); % use ylim([-40 0])

%% plot one CREW(fre) waveform power spectrum

N = 300;
sigma2 = 0.1; % noise power
sigma2jammer = 100000; % jammer power
beta = 1;
rho = 2;
s0 = golomb(N);
jammertype = 'barrage';

switch jammertype
    case 'white'
        Gamma = (sigma2 + sigma2jammer) * eye(N);
    case 'spot'
        f1 = 0.2;
        Num = 2*N-1;
        power = zeros(Num,1);
        power(floor(f1*Num)+1) = 1;
        r = ifft(power); % [r(0) r(1) ... r(N-1) r*(N-1) ... r*(1)].'
        r = r/abs(r(1)) * sigma2jammer;
        Gamma = toeplitz(r(1:N), [r(1) r(2*N-1:-1:N+1).']) ...
            + sigma2 * eye(N);
    case 'barrage'
        f1 = 0.2; f2 = 0.3;
        Num = 2*N-1;
        power = zeros(Num,1);
        power(floor(f1*Num)+1 : floor(f2*Num)) = 1;
        r = ifft(power); % [r(0) r(1) ... r(N-1) r*(N-1) ... r*(1)].'
        r = r/abs(r(1)) * sigma2jammer;
        Gamma = toeplitz(r(1:N), [r(1) r(2*N-1:-1:N+1).']) ...
            + sigma2 * eye(N);
    otherwise
        error('Unrecognized jammer type');
end

[sFRE wFRE mseFRE] = cyc_freq(N, Gamma, beta, rho, s0);
show_criterion(sFRE, Gamma, beta, 'FRE');
% save barrage_fre_300_superjamming.mat;

% myresize('barrage_2_FRE_N_300_corr_sj');
% myresize('barrage_2_FRE_N_300_seq_sj'); % use ylim([-80 0])
% myresize('barrage_2_FRE_N_300_rx_sj'); % use ylim([-80 0])

%% robust design

clear; clc;
PAR = 2;
%jammertype = 'spot';
jammertype = 'barrage';

NSet = [25 50 100 200 300 500 1000]';
%NSet = [25 50 100 200 300];
num = length(NSet);
mseTrue = zeros(num, 1);
mseFalse = zeros(num, 1);
mseRobust = zeros(num, 1);

for k = 1:num
    N = NSet(k); disp(['N = ' num2str(N)]);
    s0 = golomb(N);    
    
    [mseTrue(k) mseFalse(k) mseRobust(k)] = ...
        crew_robust(N, 'FRE', jammertype, PAR, s0);
end

figure;
semilogy(NSet, mseTrue, '-bv', NSet, mseFalse, '--bh', ...
    NSet, mseRobust, '-bh');
legend('True info', 'Wrong info', ...
    'Imprecise info with robust design');
xlabel('N');
ylabel('MSE');
title(['MSE vs. N (' ...
    jammertype ' jamming, PAR<=' num2str(PAR) ') for CREW(fre)']);
myboldify;
name = ['robust_' jammertype '_' num2str(PAR)];
myresize(name);
% save([name '.mat']);