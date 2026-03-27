rng(42);
addpath('../commons');

%% Chirp
N = 5000; 
T = 100; % total time length is 10 seconds
t = linspace(0,T,N);
B = 1; % bandwidth is 1 Hz
y = chirp(t,0,T,B);
figure; plot(t,y); 
xlabel('t (sec)'); ylabel('Real Part of s(t)');
myboldify;
myresize('chirp_shape');
y = (exp(1i * pi * B/T * t.^2)).';
r = aperplotsiso(y);
figure;
plot(linspace(-T,T,2*N-1), 20*log10(abs(r)/N));
xlabel('\tau (sec)'); ylabel('|r(\tau)/r(0)| (dB)');
axis([-T T -60 0]);
myboldify;
myresize('chirp_corr');

%% P4
N = 100;
x = p4(N);
[r isl psl] = aperplotsiso(x);
disp(['PSL: ' num2str(20*log10(psl/N))]);
figure;
plot(-N+1:N-1, 20*log10(abs(r)/N));
xlabel('k'); ylabel('|r(k)/N| (dB)');
axis([-N+1 N-1 -60 0]);
myboldify;
myresize('p4_corr');
% continuous-time waveform
tp = 1; % 1 second per pulse
T = N * tp;
sr = 100; % over-sampling rate
y = (repmat(x, [1 sr])).';
y = y(:); 
figure; 
plot(linspace(0,T,N*sr),real(y),'b-',linspace(0,T,N*sr),imag(y),'r--'); 
xlabel('t (sec)'); ylabel('s(t)');
legend('Re\{s(t)\}', 'Im\{s(t)\}');
myboldify;
myresize('p4_wave');
[r_y isl_y psl_y] = aperplotsiso(y);
figure;
plot(linspace(-T,T,2*N*sr-1), 20*log10(abs(r_y)/(N*sr)));
xlabel('\tau (sec)'); ylabel('|r(\tau)/r(0)| (dB)');
axis([-T T -60 0]);
myboldify;
myresize('p4_wave_corr');

%% m sequence
x = [1 -1 -1 1 -1 1 1]';
N = length(x);
r = perplotsiso(x);
figure;
plot(-N+1:N-1, r);
xlabel('k'); ylabel('$\tilde{r}(k)$', 'Interpreter', 'LaTex');
axis([-N+1 N-1 -3 N]);
myboldify;
myresize('m_per_corr');
r = aperplotsiso(x);
figure;
plot(-N+1:N-1, r);
xlabel('k'); ylabel('r(k)');
axis([-N+1 N-1 -3 N]);
myboldify;
myresize('m_aper_corr');

%% polyphase Barker
N = 45;
x = exp(1i * 2*pi / 90 * [0 0 7 1 76 71 76 63 56 73 87 9 9 14 ...
    25 53 62 5 32 35 85 69 40 76 57 26 9 83 56 57 21 5 52 89 48 ...
    11 68 26 62 6 37 73 19 58 12]).';
[r isl psl] = aperplotsiso(x);
disp(['PSL: ' num2str(20*log10(psl/N))]);
figure;
plot(-N+1:N-1, 20*log10(abs(r)/N));
xlabel('k'); ylabel('|r(k)/N| (dB)');
axis([-N+1 N-1 -60 0]);
myboldify;
myresize('polyphase_barker_corr');

%% Barker
x = [1 1 1 1 1 -1 -1 1 1 -1 1 -1 1]';
N = length(x);
r = aperplotsiso(x);
figure;
plot(-N+1:N-1, r);
xlabel('k'); ylabel('r(k)');
axis([-N+1 N-1 -3 N]);
myboldify;
myresize('barker_corr');

%% m-seq and Golomb
n = 7;
N = 2^n - 1;
r1 = aperplotsiso(mseq(n));
r2 = aperplotsiso(golomb(N));
figure;
plot(-(N-1):(N-1), 20*log10(abs(r1)/N), 'r--'); hold on;
plot(-(N-1):(N-1), 20*log10(abs(r2)/N), 'b-'); hold off;
xlabel('k'); ylabel('|r(k)|/N (dB)');
legend('m-seq', 'Golomb');
axis([-N+1 N-1 -80 0]);
myboldify;
myresize('mseq_Golomb');
