%% AF of Chirp
T = 10;
B = 5;
N = 500;
t = linspace(0,T,N)';
freq = B/T * t;
f_max = 2.5; % Hz, y-axis limit
caf_fcode(ones(N,1), freq, ones(N,1), freq, f_max, T);
% myresize('AF_chirp3D'); % use -depsc to color print, then open in
% photoshop, transform to gray color and save in eps 2 format
% myresize('AF_chirp2D');

%% AF of Golomb
N = 50;
x = golomb(N);
caf(x);
% myresize('AF_golomb3D');
% myresize('AF_golomb2D');

%% AF of CAN(Golomb)
N = 50;
x = cansiso(N, golomb(N));
caf(x);
% myresize('AF_can_golomb3D');
% myresize('AF_can_golomb2D');

%% AF of random
N = 50;
x0 = exp(1i*2*pi*rand(N,1));
caf(x0);
% myresize('AF_rand3D');
% myresize('AF_rand2D');

%% AF of CAN(R)
N = 50;
x = cansiso(N,x0);
caf(x);
% myresize('AF_can_rand3D');
% myresize('AF_can_rand2D');

%% AF of m-seq
N = 63; M = 6;
x = mseq(M);
caf(x);
% myresize('AF_mseq3D');
% myresize('AF_mseq2D');

%% Discrete AF example
N = 100;
x = tfca(N,[0 1/N 2/N],10);
AF = cafdis(x);
myresize('AF1-af1');

%% Wideband AF example 1
afwb(10, ones(50,1), (linspace(-1,1,50))', 40, 0.05);

%% Wideband AF example 2
% u(t) = exp(j2\pi (fc t + k t^2))
fc = 10;
N = 100; % an assumed subpulse number
k = 0.25; % 1/(sec^2)
tmin = 0; tmax = 10; % sec
f = linspace(2*k*tmin, 2*k*tmax, N);
sr = 50;
fplot_max = 0.1;
afwb(fc, ones(N,1), f', sr, fplot_max);
% myresize('AF_wb_1'); % use -depsc to color print, then open in
% photoshop, transform to gray color and save in eps 2 format
% myresize('AF_wb_2');
