%% the function that connects PAF and multi-period PAF
M = 3; T=1;
f = linspace(-2,2,200);
y = abs(sinc(f*M*T) ./ sinc(f*T));
figure;
plot(f,y);
xlabel('f');
ylabel('|sin(3\pif)/(3 sin(\pif))|');
myboldify;
myresize('sample_function');

%% PAF of Golomb
N = 50;
x = golomb(N);
cafper(x);
% myresize('PAF_golomb3D');
% myresize('PAF_golomb2D');

%% PAF of Chu
N = 50;
x = chu(N);
cafper(x);
% myresize('PAF_chu3D');
% myresize('PAF_chu2D');

%% AF of random
N = 50;
x0 = exp(1i*2*pi*rand(N,1));
cafper(x0);
% myresize('PAF_rand3D');
% myresize('PAF_rand2D');

%% AF of CAN(R)
N = 50;
x = pecansiso(N,x0);
cafper(x);
% myresize('PAF_pecan_rand3D');
% myresize('PAF_pecan_rand2D');

%% Discrete PAF example
N = 100;
x = pertfca(N,[0 1/N 2/N 3/N],15);
AF = percafdis(x, x);
myresize('PAF1-af1');
% save pertfca.mat;

