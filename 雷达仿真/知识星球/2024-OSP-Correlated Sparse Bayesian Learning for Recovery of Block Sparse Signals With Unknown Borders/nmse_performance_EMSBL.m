
%% Initialization
clear;
close all
clc
set=[30 35 40 45 50 60];
iter = 1000;
% SNR = 20;    % Signal-to-noise ratio

NMSE_set1 = zeros(length(set),iter);
NMSE_set2 = zeros(length(set),iter);
NMSE_set3 = zeros(length(set),iter);
NMSE_set4 = zeros(length(set),iter);
NMSE_set5 = zeros(length(set),iter);

jj = 1;
for m = set

    for MC = 1:iter

        n=100;                                          % signal dimension
        % m= 40;
        cp = 1;% number of measurements
        K=25;                                           % total number of nonzero coefficients
        L=3;                                            % number of nonzero blocks
        rng(MC)
        SNR =100;    % Signal-to-noise ratio
        % siga2=1e-12;
        Nsnapshot = 1;
        % generate the block-sparse signal
        x=zeros(n,1);
        r=abs(randn(L,1)); r=r+1; r=round(r*K/sum(r));
        r(L)=K-sum(r(1:L-1));                           % number of non-zero coefficients in each block
        g=round(r*n/K);
        g(L)=n-sum(g(1:L-1));
        g_cum=cumsum(g);
        for i=1:L
            % generate i-th block
            seg = 1/sqrt(2)*(randn(r(i), 1) +1i*randn(r(i),1));              % generate the non-zero block
            cp = 0.5;
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
        Phi = randn(m,n) + sqrt(-1)*randn(m,n);
        A=Phi./(ones(m,1)*sqrt(sum(Phi.^2)));
        A=Phi;
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

        %% Revoery via SBL
        eta=0;
        sigma2 =1;
        x_ma=MSBL_alternative_general(y,A,sigma2,eta,K);
        nmse1(MC)=norm(mean(abs(x_ma),2)-mean(abs(X),2))^2/org_x;

        % x_new=mu_new;
        % subplot(1,5,1)
        % nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
        %
        % stem(mean(abs(X),2),'b')
        % hold on
        % stem(mean(abs(x_new),2),'r')
        % title(['EM SBL NMSE v1: ', num2str(nmse)])

        %% Revoery via PC-SBL
        eta=1;
        sigma2 =1;
        x_pcsbla=MPCSBL_alternative(y,A,sigma2,eta);
        nmse2(MC)=norm(mean(abs(x_pcsbla),2)-mean(abs(X),2))^2/org_x;

        % x_new=mu_new;
        % subplot(1,5,2)
        % nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
        %
        % stem(mean(abs(X),2),'b')
        % hold on
        % stem(mean(abs(x_new),2),'r')
        % title(['EM PCSBL NMSE v1: ', num2str(nmse)])
        %% Revoery via PC-SBL
        eta=0;
        sigma2 =1;
        x_sbl=MPCSBL(y,A,sigma2,eta);
        % x_new=mu_new;

        % org_x = norm(mean(abs(X),2))^2;
        nmse3(MC)=norm(mean(abs(x_sbl),2)-mean(abs(X),2))^2/org_x;

        % subplot(1,6,1)
        %
        %
        % stem(mean(abs(X),2),'b')
        % hold on
        % stem(mean(abs(x_new),2),'r')
        % legend('Ground Truth','Reconstructed');
        % title(['EM SBL NMSE: ', num2str(nmse)])


        %% Revoery via PC-SBL
        eta=1;
        sigma2 =1;
        x_pcsbl=MPCSBL(y,A,sigma2,eta);
        % x_new=mu_new;
        % subplot(1,6,2)
        nmse4(MC)=norm(mean(abs(x_pcsbl),2)-mean(abs(X),2))^2/org_x;

        % stem(mean(abs(X),2),'b')
        % hold on
        % stem(mean(abs(x_new),2),'r')
        % title(['EM PCSBL NMSE: ', num2str(nmse)])


        eta=1;
        sigma2 =1;
        x_sblcorr=MSBL_correlated(y,A,sigma2,eta);
        % x_new=mu_new;
        subplot(1,5,5)
        nmse5(MC)=norm(mean(abs(x_sblcorr),2)-mean(abs(X),2))^2/org_x;


        %%
        blkLen =2;

        blkStartLoc = 1:blkLen:n;
        learnLambda = 2;
        tic;
        Result2 = BSBL_EM(A,y,blkStartLoc,learnLambda);
        x_new = Result2.x;
        subplot(2,4,4)
        nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x

        stem(mean(abs(X),2),'b')
        hold on
        stem(mean(abs(x_new),2),'r')
        title(['BSBL NMSE(h=2): ', num2str(nmse)])

        %%
        blkLen =2;
        blkStartLoc = 1:blkLen:n;
        % = 2: small noise;
        % = 0 : no noise
        tic;
        Result5 = EBSBL_BO(A, y, blkLen, noiseFlag5);
        toc;

        x_new = Result5.x;
        subplot(2,4,5)
        nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x

        stem(mean(abs(X),2),'b')
        hold on
        stem(mean(abs(x_new),2),'r')
        title(['EBSBL NMSE(h=2) : ', num2str(nmse)])
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
        subplot(2,4,6)
        nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x

        stem(mean(abs(X),2),'b')
        hold on
        stem(mean(abs(x_new),2),'r')
        title(['EBSBL NMSE(h=4)  : ', num2str(nmse)])
        %%
        blkLen =8;
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

        stem(mean(abs(X),2),'b')
        hold on
        stem(mean(abs(x_new),2),'r')
        title(['EBSBL NMSE(h=8) : ', num2str(nmse)])

        % x_new(abs(x_new)<0.4) = 0;
        % subplot(1,6,5)
        %
        % stem(mean(abs(X),2),'b')
        % hold on
        % stem(mean(abs(x_new),2),'r')
        % hold on
        % plot(abs(diag(report.results.final_iteration.gamma)),'g')
        % hold on
        % plot(abs(diag(report.results.final_iteration.gamma,1)),'p')
        % legend('ground truth','reconstructed x','diagonal \gamma','subdiagonal \gamma')
        % title(['FP TriSBL NMSE: ', num2str(nmse)])

    end

    % nmse4(nmse4>nmse3) = 0;
    NMSE_set1(jj,:) = nmse1;
    NMSE_set2(jj,:) = nmse2;
    NMSE_set3(jj,:) = nmse3;
    NMSE_set4(jj,:) = nmse4;
    NMSE_set5(jj,:) = nmse5;

    jj = jj + 1;
end

%%
figure
data = 0.01*set;

plot(data,sqrt(mean(NMSE_set1,2)),'LineWidth',3)
hold on
plot(data,sqrt(mean(NMSE_set2,2)),'LineWidth',3)
hold on
plot(data,sqrt(mean(NMSE_set3,2)),'LineWidth',3)
hold on
plot(data,sqrt(mean(NMSE_set4,2)),'LineWidth',3)
hold on
plot(data,sqrt(mean(NMSE_set5,2)),'LineWidth',3)

ylabel('NMSE')
xlabel('m/n')
legend('EM SBL v1', 'EM CSBL','EM SBL v2','EM PCSBL', 'EM Tri SBL' )
grid on
title(['K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])
title(['Correlated sources: K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])


NMSE_success_rate1 =zeros(size(NMSE_set1));
NMSE_success_rate2 =zeros(size(NMSE_set2));
NMSE_success_rate3 =zeros(size(NMSE_set3));
NMSE_success_rate4 =zeros(size(NMSE_set4));
NMSE_success_rate5 =zeros(size(NMSE_set5));

k=4;
figure
NMSE_success_rate1((NMSE_set1<10^(-k)))=1;
NMSE_success_rate2((NMSE_set2<10^(-k)))=1;
NMSE_success_rate3((NMSE_set3<10^(-k)))=1;
NMSE_success_rate4((NMSE_set4<10^(-k)))=1;
NMSE_success_rate5((NMSE_set5<10^(-k)))=1;

plot(data,mean(NMSE_success_rate1,2),'LineWidth',3)
hold on
plot(data,mean(NMSE_success_rate2,2),'LineWidth',3)
hold on
plot(data,mean(NMSE_success_rate3,2),'LineWidth',3)
hold on
plot(data,mean(NMSE_success_rate4,2),'LineWidth',3)
hold on
plot(data,mean(NMSE_success_rate5,2),'LineWidth',3)

ylabel('Success Rate')
xlabel('m/n')
legend('EM SBL v1', 'EM CSBL','EM SBL v2','EM PCSBL', 'EM Tri SBL' )
grid on
title(['K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])
title(['Correlated sources: K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])

% %%
% indices1 = find(NMSE_set5(3,:)>1);
% NMSE_set1(:,indices1) = [];
% NMSE_set2(:,indices1) = [];
% NMSE_set3(:,indices1) = [];
% NMSE_set4(:,indices1) = [];
% NMSE_set5(:,indices1) = [];
%
% indices2 = find(NMSE_set5(4,:)>1);
% NMSE_set1(:,indices2) = [];
% NMSE_set2(:,indices2) = [];
% NMSE_set3(:,indices2) = [];
% NMSE_set4(:,indices2) = [];
% NMSE_set5(:,indices2) = [];
%
%
% indices3 = find(NMSE_set5(3,:)>0.3);
% NMSE_set1(:,indices3) = [];
% NMSE_set2(:,indices3) = [];
% NMSE_set3(:,indices3) = [];
% NMSE_set4(:,indices3) = [];
% NMSE_set5(:,indices3) = [];
%
% indices4 = find(NMSE_set5(4,:)>0.3);
% NMSE_set1(:,indices4) = [];
% NMSE_set2(:,indices4) = [];
% NMSE_set3(:,indices4) = [];
% NMSE_set4(:,indices4) = [];
% NMSE_set5(:,indices4) = [];
%
% indices5 = find(NMSE_set5(5,:)>0.3);
% NMSE_set1(:,indices5) = [];
% NMSE_set2(:,indices5) = [];
% NMSE_set3(:,indices5) = [];
% NMSE_set4(:,indices5) = [];
% NMSE_set5(:,indices5) = [];
%
% indices6 = find(NMSE_set5(6,:)>0.3);
%
% NMSE_set1(:,indices6) = [];
% NMSE_set2(:,indices6) = [];
% NMSE_set3(:,indices6) = [];
% NMSE_set4(:,indices6) = [];
% NMSE_set5(:,indices6) = [];
%
% %%
% figure
% plot(data,sqrt(mean(NMSE_set1,2)),'LineWidth',3)
% hold on
% plot(data,sqrt(mean(NMSE_set2,2)),'LineWidth',3)
% hold on
% plot(data,sqrt(mean(NMSE_set3,2)),'LineWidth',3)
% hold on
% plot(data,sqrt(mean(NMSE_set4,2)),'LineWidth',3)
% hold on
% plot(data,sqrt(mean(NMSE_set5,2)),'LineWidth',3)
%
% ylabel('NMSE')
% xlabel('m/n')
% legend('EM SBL v1', 'EM CSBL','EM SBL v2','EM PCSBL', 'EM Tri SBL' )
% grid on
% title(['K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])
% title(['Correlated sources: K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])


