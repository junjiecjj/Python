%% CAN with rho>1
N = 512;
rho = 4;
x0 = exp(1i*2*pi*rand(N,1));
x1 = cansiso(N,x0);
x2 = cansisopar(rho,N,x0);

r1 = aperplotsiso(x1);
r2 = aperplotsiso(x2);

figure;
plot(-(N-1):(N-1), 20*log10(abs(r1)/N), 'b-', ...
    -(N-1):(N-1), 20*log10(abs(r2)/N), 'r.--');
xlabel('k'); ylabel('|r(k)|/N (dB)');
axis([-N+1 N-1 -140 0]);
legend('CAN, \rho=1', 'CAN, \rho=4');
myboldify;
myresize('bound_can1');

%% CAN(P4) with rho>1
N = 512;
rho = 4;
x0 = p4(N);
x1 = cansiso(N,x0);
x2 = cansisopar(rho,N,x0);

r1 = aperplotsiso(x1);
r2 = aperplotsiso(x2);

figure;
plot(-(N-1):(N-1), 20*log10(abs(r1)/N), 'b-', ...
    -(N-1):(N-1), 20*log10(abs(r2)/N), 'r.--');
xlabel('k'); ylabel('|r(k)|/N (dB)');
axis([-N+1 N-1 -140 0]);
legend('CAN(P4), \rho=1', 'CAN(P4), \rho=4');
myboldify;
myresize('bound_can2');

%% ISL vs. rho
N = 512;
rho_set = [1 1.2 1.4 1.6 1.8 2:10];
num = length(rho_set);
X = zeros(N, num); Y = zeros(N, num);
ISLx = zeros(1, num); ISLy = zeros(1, num);
x0 = exp(1i * 2*pi * rand(N,1)); y0 = p4(N);
for k = 1:num
    disp(['k = ' num2str(k)]);
    rho = rho_set(k);
    X(:,k) = cansisopar(rho, N, x0);
    Y(:,k) = cansisopar(rho, N, y0);
    [r ISLx(k)] = aperplotsiso(X(:,k));
    [r ISLy(k)] = aperplotsiso(Y(:,k));
end
figure;
semilogy(rho_set, ISLx, 'rd-', rho_set, ISLy, 'rx-');
xlabel('\rho'); ylabel('ISL');
ylim([1 10^5]);
legend('ISL of CAN', 'ISL of CAN(P4)');
myboldify;
myresize('bound_can3');
%save bound_can.mat;

%% ISL of Multi-CAN vs Bound
disp('random-phase seq');
[r ISL] = aperplotmimo(exp(1i*2*pi*rand(200,2)));
disp(['M=2,N=200,    ISL = ' num2str(ISL)]);
[r ISL] = aperplotmimo(exp(1i*2*pi*rand(512,2)));
disp(['M=2,N=512,    ISL = ' num2str(ISL)]);
[r ISL] = aperplotmimo(exp(1i*2*pi*rand(512,4)));
disp(['M=4,N=512,    ISL = ' num2str(ISL)]);
[r ISL] = aperplotmimo(exp(1i*2*pi*rand(1000,4)));
disp(['M=4,N=1000,    ISL = ' num2str(ISL)]);

disp('Multi-CAN');
[r ISL] = aperplotmimo(canmimo(200,2));
disp(['M=2,N=200,    ISL = ' num2str(ISL)]);
[r ISL] = aperplotmimo(canmimo(512,2));
disp(['M=2,N=512,    ISL = ' num2str(ISL)]);
[r ISL] = aperplotmimo(canmimo(512,4));
disp(['M=4,N=512,    ISL = ' num2str(ISL)]);
[r ISL] = aperplotmimo(canmimo(1000,4));
disp(['M=4,N=1000,    ISL = ' num2str(ISL)]);

disp('Bound');
M=2; N=200; disp(N^2 * M * (M-1));
M=2; N=512; disp(N^2 * M * (M-1));
M=4; N=512; disp(N^2 * M * (M-1));
M=4; N=1000; disp(N^2 * M * (M-1));