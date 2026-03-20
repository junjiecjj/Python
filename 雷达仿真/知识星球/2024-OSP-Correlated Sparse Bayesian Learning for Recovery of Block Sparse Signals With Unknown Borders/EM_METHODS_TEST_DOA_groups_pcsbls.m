%% Initialization
clear;
close all
clc

% Environment parameters
c = 1500;       % speed of sound
f = 200;        % frequency
lambda = c/f;   % wavelength
% ULA-horizontal array configuration
ie=1;
s=randi([1,50])
Nsnapshot = 1;
noiseFlag5 = 2;

% rng(4)
%SNR = 100;    % Signal-to-noise ratio
sigma2=1e-12;
Nsensor =40;               % number of sensors
d = 1/2*lambda;             % intersensor spacing
q = (0:1:(Nsensor-1))';     % sensor numbering
xq = (q-(Nsensor-1)/2)*d;   % sensor locations
options = SBLSet();
options.beta = 0;
n=361;                                          % signal dimension
m=Nsensor;                                           % number of measurements
K=25;                                           % total number of nonzero coefficients
L=5;
SNR =20;
% number of nonzero blocks
% generate the block-sparse signal
x=zeros(n,1);
r=abs(randn(L,1)); r=r+1; r=round(r*K/sum(r));
r(L)=K-sum(r(1:L-1));                           % number of non-zero coefficients in each block
g=round(r*n/K);
g(L)=n-sum(g(1:L-1));
%     rng(1)
% sensor configuration structure
Sensor_s.Nsensor = Nsensor;
Sensor_s.lambda = lambda;
Sensor_s.d = d;
Sensor_s.q = q;
Sensor_s.xq = xq;

% SBL options structure for version 3

options.convergence.error = 10^(-3);


% total number of snapshots

% number of random repetitions to calculate the average performance
Nsim = 1;

% range of angle space
thetalim = [-90 90];

theta_separation = 0.5;

% Bearing grid
theta = (thetalim(1):theta_separation:thetalim(2))';
Ntheta = length(theta);

% Design/steering matrix
sin_theta = sind(theta);
A = exp(-1i*2*pi/lambda*xq*sin_theta.')/sqrt(Nsensor); %.*randn(m,n) ;

Phi = randn(m,n) + sqrt(-1)*randn(m,n);
% A=Phi;
A=Phi./(ones(m,1)*sqrt(sum(Phi.^2)));
% to set Nsource parameter
options.Nsource = K;

%         rng(is, 'twister');
X=zeros(n,Nsnapshot);
r=abs(randn(L,1)); r=r+1; r=round(r*K/sum(r));
r(L)=K-sum(r(1:L-1));                           % number of non-zero coefficients in each block
g=round(r*n/K);
g(L)=n-sum(g(1:L-1));
g_cum=cumsum(g);

for i=1:L
    % generate i-th block
    seg = (1/sqrt(2)*(randn(r(i), Nsnapshot) +1j*randn(r(i),Nsnapshot)));              % generate the non-zero block
    % seg = 1*(1/sqrt(2)*(rand(r(i), Nsnapshot) +1i*rand(r(i),Nsnapshot)));              % generate the non-zero block
    %  seg = exp(1i*2*pi*randn(r(i), Nsnapshot))+1;              % generate the non-zero block
     % seg = randn(r(i), Nsnapshot);  
    
    cp = 0;
    cp2 = 0;
    R = eye(r(i)) + diag(cp*ones(r(i)-1,1),1)+diag(cp*ones(r(i)-1,1),-1)+diag(cp2*ones(r(i)-2,1),2)+diag(cp2*ones(r(i)-2,1),-2);
    seg=sqrtm(R)*(seg);
    loc=randperm(g(i)-r(i));        % the starting position of non-zero block
    x_tmp=zeros(g(i), Nsnapshot);
    x_tmp(loc(1):loc(1)-1+r(i),:)= seg;
    X(g_cum(i)-g(i)+1:g_cum(i), :)=x_tmp;
end
% cp =1;
%         x = x/max(max(x));
% noiseless measurements
% x = original_x;
% x(x~=0) = 1;
%X = repmat(x,[1 Nsnapshot]);
% X = X.*exp(1i*2*pi*randn(size(X)));

%         X = X + cp* [zeros(1,Nsnapshot); X(1:end-1,:)]+cp* [X(2:end,:); zeros(1,Nsnapshot)];
% cp = 0.8;
% cp2 = 0;
% R = eye(n) + diag(cp*ones(n-1,1),1)+diag(cp*ones(n-1,1),-1)+diag(cp2*ones(n-2,1),2)+diag(cp2*ones(n-2,1),-2);
% X=sqrtm(R)*X;

measure=A*X;

% % Observation noise, stdnoise = std(measure)*10^(-SNR/20);
% stdnoise=sqrt(sigma2);
% noise=randn(m,1)*stdnoise;
rnl = 10^(-SNR/20)*norm(X);
nwhite = complex(randn(m,1),randn(m,1))/sqrt(2*m);
noise = nwhite * rnl;	% error vector

% Noisy measurements
y=measure+noise;

org_x = norm(mean(abs(X),2))^2;


% %% Revoery via SBL
% eta=0;
% sigma2 =1;
% x_new=MSBL_alternative_general(y,A,sigma2,eta);
% % x_new=mu_new;
% subplot(2,3,1)
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
%
% stem(mean(abs(X),2),'b')
% hold on
% stem(mean(abs(x_new),2),'r')
% legend('Ground Truth','Reconstructed');
% title(['SBL NMSE v1: ', num2str(nmse)])

%% Revoery via SBL
eta=0;
sigma2 =1;
x_new=MPCSBL(y,A,sigma2,eta);
% x_new=mu_new;

subplot(4,1,1)

nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x

plot(mean(abs(X),2),'*')
hold on
stem(mean(abs(x_new),2),'r')
title(['SBL, NMSE: ', num2str(nmse)])

%% Revoery via PC-SBL
eta=1;
sigma2 =1;
x_new=MPCSBL_alternative(y,A,sigma2,eta);
% x_new=mu_new;
subplot(4,1,2)
nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x

plot(mean(abs(X),2),'*')
hold on
stem(mean(abs(x_new),2),'r')
title(['CSBL, NMSE: ', num2str(nmse)])



%% Revoery via PC-SBL
eta=1;
sigma2 =1;
x_new=MPCSBL(y,A,sigma2,eta);
% x_new=mu_new;
subplot(4,1,3)
nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x

plot(mean(abs(X),2),'*')
hold on
stem(mean(abs(x_new),2),'r')
title(['PCSBL NMSE: ', num2str(nmse)])

x_pcsbl = x_new;




% %%
% blkLen =2;
% 
% blkStartLoc = 1:blkLen:n;
% learnLambda = 2;
% tic;
% Result2 = BSBL_EM(A,y,blkStartLoc,learnLambda);
% x_new = Result2.x;
% subplot(2,4,4)
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% 
% plot(mean(abs(X),2),'*')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['BSBL(h=2), NMSE: ', num2str(nmse)])


% %%
% blkLen =4;
% 
% blkStartLoc = 1:blkLen:n;
% learnLambda = 2;
% tic;
% Result2 = BSBL_EM(A,y,blkStartLoc,learnLambda);
% x_new = Result2.x;
% subplot(2,4,5)
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% 
% plot(mean(abs(X),2),'*')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['BSBL(h=4), NMSE: ', num2str(nmse)])
% 

% 
% %%
% blkLen =2;
% blkStartLoc = 1:blkLen:n;
% % = 2: small noise;
% % = 0 : no noise
% tic;
% Result5 = EBSBL_BO(A, y, blkLen, noiseFlag5);
% toc;
% 
% x_new = Result5.x;
% subplot(2,4,6)
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% 
% plot(mean(abs(X),2),'*')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['EBSBL(h=2), NMSE: ', num2str(nmse)])
% %%
% blkLen =4;
% blkStartLoc = 1:blkLen:n;
% % = 1: strong noise;
% % = 2: small noise;
% % = 0 : no noise
% tic;
% Result5 = EBSBL_BO(A, y, blkLen, noiseFlag5);
% toc;
% 
% x_new = Result5.x;
% subplot(2,4,7)
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% 
% plot(mean(abs(X),2),'*')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['EBSBL(h=4), NMSE: ', num2str(nmse)])
% % %%
% blkLen =8;
% blkStartLoc = 1:blkLen:n;
% % = 1: strong noise;
% % = 2: small noise;
% % = 0 : no noise
% tic;
% Result5 = EBSBL_BO(A, y, blkLen, noiseFlag5);
% toc;
% 
% x_new = Result5.x;
% subplot(2,4,7)
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% 
% stem(mean(abs(X),2),'b')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['EBSBL NMSE(h=8) : ', num2str(nmse)])

%% Revoery via SBL
% figure
eta=0.5;
sigma2 =1;
x_new=MSBL_correlated_all(y,A,sigma2,eta);
% x_new=mu_new;
subplot(4,1,4)
nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x

plot(mean(abs(X),2),'*')
hold on
stem(mean(abs(x_new),2),'r')
title(['Proposed, NMSE: ', num2str(nmse)])
sgtitle(['m/n = ', num2str(m/n), ', K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])

%% Revoery via SBL

% if nmse > 0.1
% eta=1;
% sigma2 =1;
% x_new=MSBL_correlated_enhancedv2(y,A,sigma2,eta);
% % x_new=mu_new;
% figure
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% 
% plot(mean(abs(X),2),'*')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['Proposed, NMSE: ', num2str(nmse)])
% end
%
%
%
%
% Phi = [];
% x_test = [];
% for i = 1:n-1
%     Phi = [Phi A(:,i:i+1)];
%     x_test = [x_test;x(i:i+1)];
% end
%
% blkStartLoc = 1:blkLen:n;
% learnLambda = 2;
% tic;
% Result2 = BSBL_EM(Phi,y,blkStartLoc,learnLambda);
% x_con = Result2.x;
% x_new =zeros(size(x_new));

% x_new(1:2) = x_con(1:2);
% for i = 2:n-1
%     x_new = x_new + [zeros(i-1,1);x_con(i:i+1);zeros(n-i-1,1)];
% end
% subplot(2,3,5)
% nmse=norm(mean(abs(x_test),2)-mean(abs(x_con),2))^2/norm(mean(abs(x_test),2))^2
%
% stem(mean(abs(X),2),'b')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['EBSBL NMSE : ', num2str(nmse)])


% %% Revoery via SBL
% eta=1;
% sigma2 =1;
% x_new=MSBL_correlated_pcsbl(y,A,sigma2,eta);
% % x_new=mu_new;
% subplot(1,6,6)
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
%
% stem(mean(abs(X),2),'b')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['EM Tri SBL NMSE : ', num2str(nmse)])

%%
%
% figure
% imagesc(abs((x_pcsbl*x_pcsbl')))
