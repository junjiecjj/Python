%% Discrete CAF example 1: two stripes
N = 50;
g = zeros(2*N-1, N);
g(:,:) = sqrt((N^3 - N^2)/((2*N-1)*N -2*(2*N-1)));
g(:,N/2+1) = 0;
g(:,N/2+10) = 0;
g(N,N/2+1) = N;
g = sqrt(N^3) * g/norm(g,'fro');
g = g / max(g(:));
figure;
imagesc(-N+1:N-1,-N/2:N/2-1, 20*log10(g'),[-40 0]);
colormap(flipud(colormap('hot')));
set(gca,'YDir', 'normal');
xlabel('k'); ylabel('p');title('Desired Discrete CAF g_{kp}');
colorbar;
drawnow;
myboldify;
myresize('CAF2-ideal');

[x y] = ivcaf(N, g);
AF = cafdis(x, y);
myresize('CAF2-af');
disp((norm(AF,'fro'))^2);

% save DisCAF_twoStrip.mat;


%% Discrete CAF example 2, a wide stripe
N = 50;
g = zeros(2*N-1, N);
g(:,:) = sqrt((N^3 - N^2)/((2*N-1)*N -5*(2*N-1)));
g(:,(N/2-1):(N/2+3)) = 0;
g(N,N/2+1) = N;
g = sqrt(N^3) * g/norm(g,'fro');
w = ones(2*N-1, N);
w(:,(N/2-3):(N/2+5)) = 10;
g = g / max(g(:));
figure;
imagesc(-N+1:N-1,-N/2:N/2-1, 20*log10(g'),[-40 0]);
colormap(flipud(colormap('hot')));
set(gca,'YDir', 'normal');
xlabel('k'); ylabel('p');title('Desired Discrete CAF g_{kp}');
colorbar;
drawnow;
myboldify;
myresize('CAF3-ideal1');
figure;
imagesc(-N+1:N-1,-N/2:N/2-1, w');
colormap(flipud(colormap('hot')));
set(gca,'YDir', 'normal');
xlabel('k'); ylabel('p');title('Weights w_{kp}');
colorbar;
drawnow;
myboldify;
myresize('CAF3-weights1');

[x y criterion] = ivcaf_weight(N, g, w);
AF = cafdis(x, y);
myresize('CAF3-af1');
disp((norm(AF,'fro'))^2);
disp(['Energy of x: ' num2str((norm(x))^2)]);
disp(['Energy of y: ' num2str((norm(y))^2)]);
disp(['PAR of x: ' num2str(par(x))]);
figure;
semilogx(1:length(criterion), criterion, 'bx-');
xlabel('No. of Iterations');
ylabel('Criterion');
myboldify;

figure;
plot(1:N, abs(x), 'x-');
xlabel('n');
ylabel('|x(n)|');
myboldify;
myresize('CAF3-x1');

% save DisCAF_wideStrip.mat;

%% Discrete CAF example 3, a square mainlobe
N = 50;
g = zeros(2*N-1, N);
g(:,:) = sqrt((N^3 - 49*N^2)/((2*N-1)*N -49));
g(N-3:N+3,(N/2-2):(N/2+4)) = N;
g = sqrt(N^3) * g/norm(g,'fro');
w = ones(2*N-1, N);
w(N-3:N+3,(N/2-2):(N/2+4)) = 100;
g = g / max(g(:));
figure;
imagesc(-N+1:N-1,-N/2:N/2-1, 20*log10(g'),[-40 0]);
colormap(flipud(colormap('hot')));
set(gca,'YDir', 'normal');
xlabel('k'); ylabel('p');title('Desired Discrete CAF g_{kp}');
colorbar;
drawnow;
myboldify;
myresize('CAF3-ideal2');
figure;
imagesc(-N+1:N-1,-N/2:N/2-1, w');
colormap(flipud(colormap('hot')));
set(gca,'YDir', 'normal');
xlabel('k'); ylabel('p');title('Weights w_{kp}');
colorbar;
drawnow;
myboldify;
myresize('CAF3-weights2');

[x y criterion] = ivcaf_weight(N,g,w);
AF = cafdis(x, y);
myresize('CAF3-af2');
disp((norm(AF,'fro'))^2);
disp(['Energy of x: ' num2str((norm(x))^2)]);
disp(['Energy of y: ' num2str((norm(y))^2)]);
disp(['PAR of x: ' num2str(par(x))]);
figure;
semilogx(1:length(criterion), criterion, 'bx-');
xlabel('No. of Iterations');
ylabel('Criterion');
myboldify;

% save DisCAF_mainlobe.mat;


%% Regular CAF example 1: clean region close to the origin %%%%%%%
N = 50;
tp = 1;
T = N * tp;
sr = 4;
Ns = N * sr;
Nf = ceil(N/2 * sr);

t_max = 10 * tp; f_max = 2 * 1/T;
t_grid_size = tp /sr; f_grid_size = (1/T) / sr;
tm = min(floor(t_max / t_grid_size), Ns-1);
fm = min(floor(f_max / f_grid_size), Nf-1);

w = zeros(2*Nf-1, 2*Ns-1);
w((Nf-fm):(Nf+fm), (Ns-tm):(Ns+tm)) = 1;
d = zeros(2*Nf-1, 2*Ns-1);
d(Nf, Ns) = N;

[x y] = cafFit(N, tp, sr, d, w);
caf(x, y, 10);
% myresize('ex1_par_caf');
% save ex1_par.mat;
[x y] = cafFit_uni(N, tp, sr, d, w);
caf(x, y, 10);
% myresize('ex1_uni_caf');
% save ex1_uni.mat;

%% Regular CAF example 2: fit a CAF of a chirp
N = 50;
tp = 1;
T = N * tp;
sr = 2;
Ns = N * sr;
Nf = ceil(N/2 * sr);

w = ones(2*Nf-1, 2*Ns-1);
d = zeros(2*Nf-1, 2*Ns-1);
d(Nf, Ns) = N;
p1 = Nf; p2 = Nf; k1 = Ns; k2 = Ns;
for ll = 1:(Nf-1)
    p1 = p1 - 1;
    p2 = p2 + 1;
    k1 = k1 - 2;
    k2 = k2 + 2;
    d(p1, k1) = N;
    d(p2, k2) = N;
end
[x y] = cafFit(N, tp, sr, d, w);
[CAF t_grid f_grid] = caf(x, y);
% myresize('ex2_caf');

figure;
imagesc(t_grid, f_grid, d);
colormap(flipud(colormap('hot')));
set(gca, 'YDir', 'normal');
xlabel('\tau / T');
ylabel('f \times T');
title('d(\tau,f)');
colorbar;
myboldify;
myresize('ex2_d');

% save ex2.mat;