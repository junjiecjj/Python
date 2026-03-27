%% MIMO-PeCA 
N = 512;
M = 4;
P = 60;
X = pecamimo(N,M,P);
r = perplotmimo(X);
figure;
plot(-(N-1):(N-1), 20*log10(r)); hold on;
plot([-P -P], [-120 0], 'r--'); hold on;
plot([P P], [-120 0], 'r--'); hold off;
xlabel('Time Lag'); ylabel('Periodic Correlation Level');
axis([-N+1 N-1 -120 0]);
myboldify;
myresize('PeCA');
% save mimopeca.mat;

%% shift and construct
N = 512;
M = 4;
P = N/M;
x = p4(N);
X = zeros(N,M);
for k = 1:M
    X(:,k) = circshift(x, (k-1)*P);
end
r = perplotmimo(X);
figure;
plot(-(N-1):(N-1), 20*log10(r)); hold on;
xlabel('Time Lag'); ylabel('Periodic Correlation Level');
xlim([-(N-1) N-1]);
myboldify;
myresize('ZCZ');

%% Kasami vs. MIMO-PeCAN
N = 1023;
M = 4;
X1 = kasami(10); % 1023-by-32
X1 = X1(:,1:M);
X2 = pecanmimo(N,M);
X3 = exp(1i * 2*pi * rand(N,M));
[r1 ISL1 PSL1] = perplotmimo(X1); % Kasami
[r2 ISL2 PSL2] = perplotmimo(X2); % MIMO-PeCAN
[r3 ISL3 PSL3] = perplotmimo(X3); % random-phase sequence set
figure;
plot(-(N-1):(N-1), 20*log10(r1));
xlabel('Time Lag'); ylabel('Periodic Correlation Level');
axis([-(N-1) N-1 -30 0]);
myboldify;
myresize('Kasami');
figure;
plot(-(N-1):(N-1), 20*log10(r2));
xlabel('Time Lag'); ylabel('Periodic Correlation Level');
axis([-(N-1) N-1 -30 0]);
myboldify;
myresize('MIMOPeCAN');
figure;
plot(-(N-1):(N-1), 20*log10(r3));
xlabel('Time Lag'); ylabel('Periodic Correlation Level');
axis([-(N-1) N-1 -30 0]);
myboldify;
myresize('MIMOrandomphase');
disp(['Kasami: ISL -- ' num2str(ISL1) ', PSL -- ' num2str(PSL1)]);
disp(['MIMO-PeCAN: ISL -- ' num2str(ISL2) ', PSL -- ' num2str(PSL2)]);
disp(['Random-phase: ISL -- ' num2str(ISL3) ', PSL -- ' num2str(PSL3)]);