

%% Initialization
clear;
close all
clc

n=100;                                          % signal dimension
m= 35;
cp = 1;% number of measurements
K=25;                                           % total number of nonzero coefficients
L=5;                                            % number of nonzero blocks
SNR = 100;    % Signal-to-noise ratio
noiseFlag5 = 2;
learnLambda = 0;

% siga2=1e-12;
Nsnapshot = 1;
% load DWTM.mat
% A=ww;
% rng(2)
% generate the block-sparse signal
% rng(20)
x=zeros(n,1);
r=abs(randn(L,1)); 
r=r+1;
r=round(r*K/sum(r));
r(L)=K-sum(r(1:L-1));                           % number of non-zero coefficients in each block
g=round(r*n/K);
g(L)=n-sum(g(1:L-1));
g_cum=cumsum(g);
for i=1:L
    % generate i-th block
    seg = 1/sqrt(2)*(randn(r(i), 1) +1j*randn(r(i),1));              % generate the non-zero bloc
    % seg = abs(randn(r(i), 1));
    cp = 0;
    cp2 = 0;
    R = eye(r(i)) + diag(cp*ones(r(i)-1,1),1)+diag(cp*ones(r(i)-1,1),-1)+diag(cp2*ones(r(i)-2,1),2)+diag(cp2*ones(r(i)-2,1),-2);
    seg=sqrtm(R)*seg;
    loc=randperm(g(i)-r(i));        % the starting position of non-zero block
    x_tmp=zeros(g(i), 1);
    x_tmp(loc(1):loc(1)-1+r(i))= seg;
    x(g_cum(i)-g(i)+1:g_cum(i), 1)=x_tmp;
end

% generate the measurement matrix
Phi=randn(m,n);
% Phi = randn(m,n) + sqrt(-1)*randn(m,n);
% A=Phi;
A=Phi./(ones(m,1)*sqrt(sum(Phi.^2)));

% A = dftmtx(m,n);
% noiseless measurements
% x(x~=0) = 1;
% cp=0.8;
X = repmat(x,[1 Nsnapshot]);
% X = X.*exp(1i*2*pi*randn(size(X)));
% %

% X(abs(X) < 0.3) = 0;
% X(80,:) = 1;
% x(80) =1;





measure=A*X;

% % Observation noise, stdnoise = std(measure)*10^(-SNR/20);
% stdnoise=sqrt(sigma2);
% noise=randn(m,1)*stdnoise;
rnl = 10^(-SNR/20)*norm(x);
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
figure
eta=0;
sigma2 =1;
x_new=MPCSBL(y,A,sigma2,eta);
% x_new=mu_new;

subplot(2,4,1)

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
subplot(2,4,2)
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
subplot(2,4,3)
nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x

plot(mean(abs(X),2),'*')
hold on
stem(mean(abs(x_new),2),'r')
title(['PCSBL NMSE: ', num2str(nmse)])

x_pcsbl = x_new;




%%
blkLen =2;

blkStartLoc = 1:blkLen:n;
tic;
Result2 = BSBL_EM(A,y,blkStartLoc,learnLambda);
x_new = Result2.x;
subplot(2,4,4)
nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x

plot(mean(abs(X),2),'*')
hold on
stem(mean(abs(x_new),2),'r')
title(['BSBL(h=2), NMSE: ', num2str(nmse)])


%%
blkLen =4;

blkStartLoc = 1:blkLen:n;
tic;
Result2 = BSBL_EM(A,y,blkStartLoc,learnLambda);
x_new = Result2.x;
subplot(2,4,5)
nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x

plot(mean(abs(X),2),'*')
hold on
stem(mean(abs(x_new),2),'r')
title(['BSBL(h=4), NMSE: ', num2str(nmse)])



%%
blkLen =2;
blkStartLoc = 1:blkLen:n;
% = 2: small noise;
% = 0 : no noise
tic;
Result5 = EBSBL_BO(A, y, blkLen, noiseFlag5);
toc;

x_new = Result5.x;
subplot(2,4,6)
nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x

plot(mean(abs(X),2),'*')
hold on
stem(mean(abs(x_new),2),'r')
title(['EBSBL(h=2), NMSE: ', num2str(nmse)])
%%
blkLen =4;
blkStartLoc = 1:blkLen:n;
% = 1: strong noise;
% = 2: small noise;
% = 0 : no noise
tic;
Result5 = EBSBL_BO(A, y, blkLen, noiseFlag5);
toc;

x_new = Result5.x;
subplot(2,4,7)
nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x

plot(mean(abs(X),2),'*')
hold on
stem(mean(abs(x_new),2),'r')
title(['EBSBL(h=4), NMSE: ', num2str(nmse)])
% %%
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
figure
eta=0.5;
sigma2 =1;
[x_new0,EM_variable,alpha_new0]=MSBL_correlated_all(y,A,sigma2,eta);
% x_new=mu_new;
% subplot(1,3,1)
nmse=norm(mean(abs(x_new0),2)-mean(abs(X),2))^2/org_x

ll = 2
figure
plot(EM_variable,'LineWidth',ll)
title('-0.5log(|\Gamma|)-0.5tr(\Gamma^{-1}(\Sigma_s + \mu_s \mu_s^H))')
hold on
eta=0.1;
sigma2 =1;
[x_new01,EM_variable,alpha_new01]=MSBL_correlated_all(y,A,sigma2,eta);

% x_new=mu_new;
% subplot(1,3,1)
nmse01=norm(mean(abs(x_new01),2)-mean(abs(X),2))^2/org_x
plot(real(EM_variable),'LineWidth',ll)
hold on
eta=0.2;
sigma2 =1;
[x_new02,EM_variable,alpha_new02]=MSBL_correlated_all(y,A,sigma2,eta);
% x_new=mu_new;
% subplot(1,3,1)
nmse02=norm(mean(abs(x_new02),2)-mean(abs(X),2))^2/org_x
plot(real(EM_variable),'LineWidth',ll)
hold on
eta=0.3;
sigma2 =1;
[x_new03,EM_variable,alpha_new03]=MSBL_correlated_all(y,A,sigma2,eta);
% x_new=mu_new;
% subplot(1,3,1)
nmse03=norm(mean(abs(x_new03),2)-mean(abs(X),2))^2/org_x
plot(real(EM_variable),'LineWidth',ll)
hold on
eta=0.4;
sigma2 =1;
[x_new04,EM_variable,alpha_new04]=MSBL_correlated_all(y,A,sigma2,eta);
% x_new=mu_new;
% subplot(1,3,1)
nmse04=norm(mean(abs(x_new04),2)-mean(abs(X),2))^2/org_x
plot(real(EM_variable),'LineWidth',ll)
hold on
eta=0.5;
sigma2 =1;
[x_new05,EM_variable,alpha_new05]=MSBL_correlated_all(y,A,sigma2,eta);
% x_new=mu_new;
% subplot(1,3,1)
nmse05=norm(mean(abs(x_new05),2)-mean(abs(X),2))^2/org_x
plot(real(EM_variable),'LineWidth',ll)
hold on
grid on
legend('SBL','Proposed \beta=0.1','Proposed \beta=0.2','Proposed \beta=0.3','Proposed \beta=0.4','Proposed \beta=0.5','Proposed \beta=0.6','Proposed \beta=0.7','Proposed \beta=0.8','Proposed \beta=0.9','Proposed \beta=1')


figure
subplot(2,3,1)
plot(mean(abs(X),2),'*')
title(['SBL, NMSE: ', num2str(nmse)])
 hold on
stem(mean(abs(x_new0),2),'r')
legend('ground truth','reconstructed')
subplot(2,3,2)
plot(mean(abs(X),2),'*')
title(['Proposed \beta =0.1, NMSE: ', num2str(nmse01)])
hold on
stem(mean(abs(x_new01),2),'r')

subplot(2,3,3)
plot(mean(abs(X),2),'*')
hold on
title(['Proposed \beta =0.2, NMSE: ', num2str(nmse02)])
stem(mean(abs(x_new02),2),'r')

subplot(2,3,4)
plot(mean(abs(X),2),'*')
hold on
title(['Proposed \beta =0.3, NMSE: ', num2str(nmse03)])
stem(mean(abs(x_new03),2),'r')

subplot(2,3,5)
hold on
plot(mean(abs(X),2),'*')
title(['Proposed \beta =0.4, NMSE: ', num2str(nmse04)])
stem(mean(abs(x_new04),2),'r')

subplot(2,3,6)
plot(mean(abs(X),2),'*')
hold on
stem(mean(abs(x_new05),2),'r')
title(['Proposed \beta =0.5, NMSE: ', num2str(nmse05)])

% eta=0.6;
% sigma2 =1;
% [x_new,EM_variable,alpha_new]=MSBL_correlated_all(y,A,sigma2,eta);
% % x_new=mu_new;
% % subplot(1,3,1)
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% plot(EM_variable,'LineWidth',ll)
% hold on
% eta=0.7;
% sigma2 =1;
% [x_new,EM_variable,alpha_new]=MSBL_correlated_all(y,A,sigma2,eta);
% % x_new=mu_new;
% % subplot(1,3,1)
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% plot(EM_variable,'LineWidth',ll)
% hold on
% eta=0.8;
% sigma2 =1;
% [x_new,EM_variable,alpha_new]=MSBL_correlated_all(y,A,sigma2,eta);
% % x_new=mu_new;
% % subplot(1,3,1)
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% plot(EM_variable,'LineWidth',ll)
% hold on
% eta=0.9;
% sigma2 =1;
% [x_new,EM_variable,alpha_new]=MSBL_correlated_all(y,A,sigma2,eta);
% % x_new=mu_new;
% % subplot(1,3,1)
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% plot(EM_variable,'LineWidth',ll)
% hold on
% eta=1;
% sigma2 =1;
% [x_new,EM_variable,alpha_new]=MSBL_correlated_all(y,A,sigma2,eta);
% % x_new=mu_new;
% % subplot(1,3,1)
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% plot(EM_variable,'LineWidth',ll)
% hold on
% % eta=0.5;
% % sigma2 =1;
% [x_new,EM_variable,alpha_new]=MSBL_correlated_full(y,A,sigma2,eta);
% % x_new=mu_new;
% % subplot(1,3,1)
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% plot(EM_variable,'LineWidth',ll)
% 
% title(['SNR :', num2str(SNR),'dB'])


% figure
% plot(mean(abs(X),2),'*')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['SBL, NMSE: ', num2str(nmse)])
% sgtitle(['m/n = ', num2str(m/n), ', K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])

% %% Revoery via SBL
% eta=0.5;
% sigma2 =1;
% x_new=MSBL_correlated(y,A,sigma2,eta);
% % x_new=mu_new;
% figure
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% 
% plot(mean(abs(X),2),'*')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['Proposed2, NMSE: ', num2str(nmse)])
% % sgtitle(['m/n = ', num2str(m/n), ', K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])
% 
%% Revoery via SBL
% 
% figure
% 
% inner_it=100;
% subplot(1,4,1)
% % if nmse > 0.1
% % eta=1;
% % sigma2 =1;
% % x_new=MSBL_correlated_enhanced(y,A,sigma2,eta,inner_it);
% % x_new=mu_new;
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% 
% plot(mean(abs(X),2),'*')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['Proposed, NMSE: ', num2str(nmse)])
% % sgtitle(['m/n = ', num2str(m/n), ', K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])
% 
% % inner_it=10;
% subplot(1,4,2)
% % if nmse > 0.1
% eta=0.5;
% sigma2 =1;
% x_new=MSBL_correlated_enhanced(y,A,sigma2,eta,inner_it);
% % x_new=mu_new;
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% 
% plot(mean(abs(X),2),'*')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['Proposed GD, NMSE: ', num2str(nmse)])
% 
% 
% subplot(1,4,3)
% % if nmse > 0.1
% eta=0.5;
% sigma2 =1;
% x_new=MSBL_correlated_enhanced_fp(y,A,sigma2,eta,inner_it);
% % x_new=mu_new;
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% 
% plot(mean(abs(X),2),'*')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['Proposed FP, NMSE: ', num2str(nmse)])
% 
% subplot(1,4,4)
% % if nmse > 0.1
% eta=0.5;
% sigma2 =1;
% x_new=MSBL_correlated_enhanced_fpv2(y,A,sigma2,eta,inner_it);
% % x_new=mu_new;
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% 
% plot(mean(abs(X),2),'*')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['Proposed FP2, NMSE: ', num2str(nmse)])
% % end
% %
% %
% %
% %
% % Phi = [];
% % x_test = [];
% % for i = 1:n-1
% %     Phi = [Phi A(:,i:i+1)];
% %     x_test = [x_test;x(i:i+1)];
% % end
% %
% % blkStartLoc = 1:blkLen:n;
% % learnLambda = 2;
% % tic;
% % Result2 = BSBL_EM(Phi,y,blkStartLoc,learnLambda);
% % x_con = Result2.x;
% % x_new =zeros(size(x_new));
% 
% % x_new(1:2) = x_con(1:2);
% % for i = 2:n-1
% %     x_new = x_new + [zeros(i-1,1);x_con(i:i+1);zeros(n-i-1,1)];
% % end
% % subplot(2,3,5)
% % nmse=norm(mean(abs(x_test),2)-mean(abs(x_con),2))^2/norm(mean(abs(x_test),2))^2
% %
% % stem(mean(abs(X),2),'b')
% % hold on
% % stem(mean(abs(x_new),2),'r')
% % title(['EBSBL NMSE : ', num2str(nmse)])
% 
% 
% % %% Revoery via SBL
% % eta=1;
% % sigma2 =1;
% % x_new=MSBL_correlated_pcsbl(y,A,sigma2,eta);
% % % x_new=mu_new;
% % subplot(1,6,6)
% % nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
% %
% % stem(mean(abs(X),2),'b')
% % hold on
% % stem(mean(abs(x_new),2),'r')
% % title(['EM Tri SBL NMSE : ', num2str(nmse)])
% 
% %%
% %
% % figure
% % imagesc(abs((x_pcsbl*x_pcsbl')))
