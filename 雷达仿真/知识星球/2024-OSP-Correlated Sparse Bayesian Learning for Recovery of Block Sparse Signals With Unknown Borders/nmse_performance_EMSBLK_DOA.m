

%% Initialization
clear;
close all
clc
set=[12 16 20 24 28];
iter = 100;
SNR =10;
% Environment parameters
c = 1500;       % speed of sound
f = 200;        % frequency
lambda = c/f;   % wavelength

NMSE_set_sbl = zeros(length(set),iter);
NMSE_set_csbl = zeros(length(set),iter);
NMSE_set_pcsbl = zeros(length(set),iter);
NMSE_set_bsbl2 = zeros(length(set),iter);
NMSE_set_bsbl4 = zeros(length(set),iter);
NMSE_set_ebsbl2 = zeros(length(set),iter);
NMSE_set_ebsbl4 = zeros(length(set),iter);
NMSE_set_corr = zeros(length(set),iter);
NMSE_set_corr2 = zeros(length(set),iter);


nmse_bsbl2 = zeros(1,iter);
nmse_bsbl4 = zeros(1,iter);
nmse_ebsbl2 = zeros(1,iter);
nmse_ebsbl4 = zeros(1,iter);


sup_set_sbl = zeros(length(set),iter);
sup_set_csbl = zeros(length(set),iter);
sup_set_pcsbl = zeros(length(set),iter);
sup_set_bsbl2 = zeros(length(set),iter);
sup_set_ebsbl4 = zeros(length(set),iter);
sup_set_ebsbl2 = zeros(length(set),iter);
sup_set_bsbl4 = zeros(length(set),iter);
sup_set_corr = zeros(length(set),iter);
sup_set_corr2 = zeros(length(set),iter);

rat = 0.05;
jj = 1;
for K = set

    for MC = 1:iter
        rng(MC)
        ie=1;
        % s=randi([1,50])
        Nsnapshot = 1;
        noiseFlag5 = 2;
        Nsensor = 45;
        % rng(50)
        %SNR = 100;    % Signal-to-noise ratio
        sigma2=1e-12;
        % Nsensor =40;               % number of sensors
        d = 1/2*lambda;             % intersensor spacing
        q = (0:1:(Nsensor-1))';     % sensor numbering
        xq = (q-(Nsensor-1)/2)*d;   % sensor locations
        options = SBLSet();
        options.beta = 0;
        n=361;                                          % signal dimension
        m=Nsensor;                                           % number of measurements
        % K=25;                                           % total number of nonzero coefficients
        L=5;        % number of nonzero blocks
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

            cp = 0.5;
            cp2 = 0;
            R = eye(r(i)) + diag(cp*ones(r(i)-1,1),1)+diag(cp*ones(r(i)-1,1),-1);%+diag(cp2*ones(r(i)-2,1),2)+diag(cp2*ones(r(i)-2,1),-2);
            seg=sqrtm(R)*(seg);
            loc=randperm(g(i)-r(i));        % the starting position of non-zero block
            x_tmp=zeros(g(i), Nsnapshot);
            x_tmp(loc(1):loc(1)-1+r(i),:)= seg;
            X(g_cum(i)-g(i)+1:g_cum(i), :)=x_tmp;
        end



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

        %% Revoery via PC-SBL
        eta=1;
        sigma2 =1;
        x_csbl=MPCSBL_alternative(y,A,sigma2,eta);
        nmse_csbl(MC)=norm(mean(abs(x_csbl),2)-mean(abs(X),2))^2/org_x;

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
        nmse_sbl(MC)=norm(mean(abs(x_sbl),2)-mean(abs(X),2))^2/org_x;

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
        nmse_pcsbl(MC)=norm(mean(abs(x_pcsbl),2)-mean(abs(X),2))^2/org_x;

        % stem(mean(abs(X),2),'b')
        % hold on
        % stem(mean(abs(x_new),2),'r')
        % title(['EM PCSBL NMSE: ', num2str(nmse)])


        eta=0.5;
        sigma2 =1;
        x_sblcorr=MSBL_correlated(y,A,sigma2,eta);
        % x_new=mu_new;
        % subplot(1,5,5)
        nmse_corr(MC)=norm(mean(abs(x_sblcorr),2)-mean(abs(X),2))^2/org_x;


        eta=0.5;
        sigma2 =1;
        x_sblcorr2=MSBL_correlated_second(y,A,sigma2,eta);
        % x_new=mu_new;
        % subplot(1,5,5)
        nmse_corr2(MC)=norm(mean(abs(x_sblcorr2),2)-mean(abs(X),2))^2/org_x;
        % %%
        % blkLen =2;
        % 
        % blkStartLoc = 1:blkLen:n;
        % learnLambda = 2;
        % Result2 = BSBL_EM(A,y,blkStartLoc,learnLambda);
        % x_bsbl2 = Result2.x;
        % nmse_bsbl2(MC)=norm(mean(abs(x_bsbl2),2)-mean(abs(X),2))^2/org_x;
        % 
        % 
        % %%
        % blkLen =4;
        % 
        % blkStartLoc = 1:blkLen:n;
        % learnLambda = 2;
        % Result2 = BSBL_EM(A,y,blkStartLoc,learnLambda);
        % x_bsbl4 = Result2.x;
        % nmse_bsbl4(MC)=norm(mean(abs(x_bsbl4),2)-mean(abs(X),2))^2/org_x;
        % 
        % %%
        % blkLen =2;
        % blkStartLoc = 1:blkLen:n;
        % % = 2: small noise;
        % % = 0 : no noise
        % Result5 = EBSBL_BO(A, y, blkLen, noiseFlag5);
        % 
        % x_ebsbl2 = Result5.x;
        % nmse_ebsbl2(MC)=norm(mean(abs(x_ebsbl2),2)-mean(abs(X),2))^2/org_x;
        % %%
        % blkLen =4;
        % blkStartLoc = 1:blkLen:n;
        % % = 1: strong noise;
        % % = 2: small noise;
        % % = 0 : no noise
        % Result4 = EBSBL_BO(A, y, blkLen, noiseFlag5);
        % 
        % x_ebsbl4 = Result4.x;
        % nmse_ebsbl4(MC)=norm(mean(abs(x_ebsbl4),2)-mean(abs(X),2))^2/org_x;

        sup_X = zeros(size(X));
        sup_x_sbl = zeros(size(X));
        sup_x_csbl = zeros(size(X));
        sup_x_pcsbl = zeros(size(X));
        sup_x_bsbl2 = zeros(size(X));
        sup_x_bsbl4 = zeros(size(X));
        sup_x_ebsbl2 = zeros(size(X));
        sup_x_ebsbl4 = zeros(size(X));
        sup_x_corr = zeros(size(X));

        sup_X(abs(X)>rat) = 1;
        sup_x_sbl(abs(x_sbl) > rat)=1;
        sup_x_csbl(abs(x_csbl) > rat)=1;
        sup_x_pcsbl(abs(x_pcsbl) > rat)=1;
        % sup_x_bsbl2(abs(x_bsbl2) > rat)=1;
        % sup_x_bsbl4(abs(x_bsbl4) > rat)=1;
        % sup_x_ebsbl2(abs(x_ebsbl2) > rat)=1;
        % sup_x_ebsbl4(abs(x_ebsbl4) > rat)=1;
        sup_x_corr(abs(x_sblcorr) > rat)=1;

        sum_temp = zeros(size(X));
        sum_temp((sup_X+sup_x_sbl)==2) =1;
        srate_sbl(MC) = sum(sum_temp)/(sum(abs(sup_x_sbl-sup_X)) + sum(abs(sup_X)));

        sum_temp = zeros(size(X));
        sum_temp((sup_X+sup_x_csbl)==2) =1;
        srate_csbl(MC) = sum(sum_temp)/(sum(abs(sup_x_csbl-sup_X)) + sum(abs(sup_X)));

        sum_temp = zeros(size(X));
        sum_temp((sup_X+sup_x_pcsbl)==2) =1;
        srate_pcsbl(MC) = sum(sum_temp)/(sum(abs(sup_x_pcsbl-sup_X)) + sum(abs(sup_X)));


        sum_temp = zeros(size(X));
        sum_temp((sup_X+sup_x_corr)==2) =1;
        srate_corr(MC)= sum(sum_temp)/(sum(abs(sup_x_corr-sup_X)) + sum(abs(sup_X)));

        % sum_temp = zeros(size(X));
        % sum_temp((sup_X+sup_x_bsbl2)==2) =1;
        % srate_bsbl2(MC) = sum(sum_temp)/(sum(abs(sup_x_bsbl2-sup_X)) + sum(abs(sup_X)));
        % 
        % sum_temp = zeros(size(X));
        % sum_temp((sup_X+sup_x_bsbl4)==2) =1;
        % srate_bsbl4(MC)= sum(sum_temp)/(sum(abs(sup_x_bsbl4-sup_X)) + sum(abs(sup_X)));
        % 
        % sum_temp = zeros(size(X));
        % sum_temp((sup_X+sup_x_ebsbl2)==2) =1;
        % srate_ebsbl2(MC) = sum(sum_temp)/(sum(abs(sup_x_ebsbl2-sup_X)) + sum(abs(sup_X)));
        % 
        % sum_temp = zeros(size(X));
        % sum_temp((sup_X+sup_x_ebsbl4)==2) =1;
        % srate_ebsbl4(MC) = sum(sum_temp)/(sum(abs(sup_x_ebsbl4-sup_X)) + sum(abs(sup_X)));

    end

    % nmse4(nmse4>nmse3) = 0;
    NMSE_set_csbl(jj,:) = nmse_csbl;
    NMSE_set_sbl(jj,:) = nmse_sbl;
    NMSE_set_pcsbl(jj,:) = nmse_pcsbl;
    NMSE_set_corr(jj,:) = nmse_corr;
    NMSE_set_corr2(jj,:) = nmse_corr2;
    NMSE_set_bsbl2(jj,:) = nmse_bsbl2;
    NMSE_set_bsbl4(jj,:) = nmse_bsbl4;
    NMSE_set_ebsbl2(jj,:) = nmse_ebsbl2;
    NMSE_set_ebsbl4(jj,:) = nmse_ebsbl4;


    % nmse4(nmse4>nmse3) = 0;
    sup_set_csbl(jj,:) = srate_csbl;
    sup_set_sbl(jj,:) = srate_sbl;
    sup_set_pcsbl(jj,:) = srate_pcsbl;
    sup_set_corr(jj,:) = srate_corr;
    % sup_set_bsbl2(jj,:) = srate_bsbl2;
    % sup_set_bsbl4(jj,:) = srate_bsbl4;
    % sup_set_ebsbl2(jj,:) = srate_ebsbl2;
    % sup_set_ebsbl4(jj,:) = srate_ebsbl4;
    jj = jj + 1;
end

%%

sup_set_sbl(sup_set_sbl<1)=0;
sup_set_csbl(sup_set_csbl<1)=0;
sup_set_pcsbl(sup_set_pcsbl<1)=0;
sup_set_bsbl2=0;
sup_set_bsbl4=0;
sup_set_ebsbl2=0;
sup_set_ebsbl4=0;
sup_set_corr(sup_set_corr<1)=0;




data = set/361;
ll=2;
plot(data,(mean(sup_set_sbl,2)),'LineWidth',ll)
hold on
plot(data,(mean(sup_set_csbl,2)),'LineWidth',ll)
hold on
plot(data,(mean(sup_set_pcsbl,2)),'LineWidth',ll)
hold on
% plot(data,(mean(sup_set_bsbl2,2)),'LineWidth',ll)
% hold on
% plot(data,(mean(sup_set_bsbl4,2)),'LineWidth',ll)
% hold on
% plot(data,(mean(sup_set_ebsbl2,2)),'LineWidth',ll)
% hold on
% plot(data,(mean(sup_set_ebsbl4,2)),'LineWidth',ll)
hold on
plot(data,(mean(sup_set_corr,2)),'--*','LineWidth',ll)

ylabel('Support Recovery Rate')
xlabel('m/n')
legend('SBL', 'CSBL','PCSBL', 'Proposed' )
grid on
title(['Correlated sources: K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])


figure
data = set;
ll=2;
plot(data,(mean(NMSE_set_sbl,2)),'LineWidth',ll)
hold on
plot(data,(mean(NMSE_set_csbl,2)),'LineWidth',ll)
hold on
plot(data,(mean(NMSE_set_pcsbl,2)),'LineWidth',ll)
hold on
% plot(data,(mean(NMSE_set_bsbl2,2)),'LineWidth',ll)
% hold on
% plot(data,(mean(NMSE_set_bsbl4,2)),'LineWidth',ll)
% hold on
% plot(data,(mean(NMSE_set_ebsbl2,2)),'LineWidth',ll)
% hold on
% plot(data,(mean(NMSE_set_ebsbl4,2)),'LineWidth',ll)
% hold on
plot(data,(mean(NMSE_set_corr,2)),'--*','LineWidth',ll)
% hold on
% plot(data,(mean(NMSE_set_corr2,2)),'--*','LineWidth',ll)

ylabel('NMSE')
xlabel('Sparsity level (K)')
legend('SBL', 'CSBL','PCSBL', 'Proposed' )
grid on
title(['Uncorrelated sources: m = ', num2str(m),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])

NMSE_sr_bsbl2 =zeros(size(NMSE_set_bsbl2));
NMSE_sr_bsbl4 =zeros(size(NMSE_set_bsbl4));
NMSE_sr_ebsbl2 =zeros(size(NMSE_set_ebsbl2));
NMSE_sr_ebsbl4 =zeros(size(NMSE_set_ebsbl4));
NMSE_sr_csbl =zeros(size(NMSE_set_csbl));
NMSE_sr_sbl =zeros(size(NMSE_set_sbl));
NMSE_sr_pcsbl =zeros(size(NMSE_set_pcsbl));
NMSE_sr_corr =zeros(size(NMSE_set_corr));
NMSE_sr_corr2 =zeros(size(NMSE_set_corr2));

figure
k=3;

NMSE_sr_bsbl2((NMSE_set_bsbl2<10^(-k)))=1;
NMSE_sr_bsbl4((NMSE_set_bsbl4<10^(-k)))=1;
NMSE_sr_ebsbl2((NMSE_set_ebsbl2<10^(-k)))=1;
NMSE_sr_ebsbl4((NMSE_set_ebsbl4<10^(-k)))=1;
NMSE_sr_csbl((NMSE_set_csbl<10^(-k)))=1;
NMSE_sr_sbl((NMSE_set_sbl<10^(-k)))=1;
NMSE_sr_pcsbl((NMSE_set_pcsbl<10^(-k)))=1;
NMSE_sr_corr((NMSE_set_corr<10^(-k)))=1;
NMSE_sr_corr2((NMSE_set_corr2<10^(-k)))=1;


plot(data,mean(NMSE_sr_sbl,2),'LineWidth',ll)
hold on
plot(data,mean(NMSE_sr_csbl,2),'LineWidth',ll)
hold on
plot(data,mean(NMSE_sr_pcsbl,2),'LineWidth',ll)
hold on
% plot(data,mean(NMSE_sr_bsbl2,2),'LineWidth',ll)
% hold on
% plot(data,mean(NMSE_sr_bsbl4,2),'LineWidth',ll)
% hold on
% plot(data,mean(NMSE_sr_ebsbl2,2),'LineWidth',ll)
% hold on
% plot(data,mean(NMSE_sr_ebsbl4,2),'LineWidth',ll)
hold on
plot(data,mean(NMSE_sr_corr,2),'--*','LineWidth',ll)
% hold on
% plot(data,mean(NMSE_sr_corr2,2),'--*','LineWidth',ll)

ylabel('Success Rate')
xlabel('m')
legend('SBL', 'CSBL','PCSBL','Proposed' )
grid on
title(['Correlated sources: K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])





% %%
%
% %%
% x_real = X;
% % For SR using pFISTA_diag_US
% cs_params.Beta       = 1;
% cs_params.L0         = [];         % Lipschitz constant
% cs_params.LambdaBar  = 1e-7;
% cs_params.Lambda     = 10^7;        % 1 - l1 regularization parameters. 0.0001 for TV, 0.009 for analysis
% cs_params.IterMax    = 150;
% cs_params.NonNegOrth = 0;
% cs_params.MaxTimeLag = 1;
% cs_params.sizeY = size(x_real);
% cs_params.Lambda= 8*10^1;        % 1 - l1 regularization parameters. 0.0001 for TV, 0.009 for analysis
% X_out_all = pFISTA_diag_US_ABZ_alternativeI_noconvA( y, A, cs_params );
% if size(X_out_all,3) > 1
%     X_out = squeeze(X_out_all);
%     X_output = reshape(X_out,size(x_real,1),size(x_real,2),size(X_out,2));
% else
%     X_out = squeeze(X_out_all);
%     X_output = reshape(X_out,size(x_real,1),size(x_real,2),size(X_out,3));
%
% end
% % x_ref = mean(abs(x_real)/max(max(abs(x_real))),3);
%
% x_mmv = double(mean(abs(X_output),3));
%
% subplot(1,6,6)
% nmse=norm(mean(abs(x_mmv),2)-mean(abs(X),2))^2/org_x
%
% stem(mean(abs(X),2),'b')
% hold on
% stem(mean(abs(x_mmv),2),'r')
%
% title(['MMV FISTA NMSE: ', num2str(nmse)])
%
% sgtitle(['m/n = ', num2str(m/n), ', K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])


% %%
% x_new = A'*y;
% x_new = reshape(x_new,size(X));
%
% % x_new(abs(x_new)>1) = 1;
% figure
% nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
%
% stem(abs(x),'b')
% hold on
% stem(mean(abs(x_new),2),'r')
% title(['FP TriSBL NMSE: ', num2str(nmse)])
% figure
% imagesc(abs((X*X')))

