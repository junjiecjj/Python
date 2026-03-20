

%% Initialization
clear;
close all
clc
set=[8 12 16 20 24 28];
iter = 1;
iter_in = 50;
SNR = 100;    % Signal-to-noise ratio

NMSE_set_sbl = zeros(length(set),iter);
NMSE_set_corr = zeros(length(set),iter);
NMSE_set_corrgd = zeros(length(set),iter);
NMSE_set_corrfp = zeros(length(set),iter);
NMSE_set_corrfp2 = zeros(length(set),iter);

sup_set_sbl = zeros(length(set),iter);
sup_set_corr = zeros(length(set),iter);
sup_set_corrgd = zeros(length(set),iter);
sup_set_corrfp = zeros(length(set),iter);
sup_set_corrfp2 = zeros(length(set),iter);

rat = 0.05;
jj = 1;
for K= set

    for MC = 1:iter

        n=100;  
        m=40;% signal dimension
        cp = 1;% number of measurements
        % K=25;                                           % total number of nonzero coefficients
        L=3;                                            % number of nonzero blocks
        rng(MC)
        if SNR > 25
            noiseFlag5 = 0;
        else 
            noiseFlag5 = 2;
        end
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
            R = eye(r(i)) + diag(cp*ones(r(i)-1,1),1)+diag(cp*ones(r(i)-1,1),-1); %+diag(cp2*ones(r(i)-2,1),2)+diag(cp2*ones(r(i)-2,1),-2);
            seg=sqrtm(R)*seg;
            loc=randperm(g(i)-r(i));        % the starting position of non-zero block
            x_tmp=zeros(g(i), 1);
            x_tmp(loc(1):loc(1)-1+r(i))= seg;
            x(g_cum(i)-g(i)+1:g_cum(i), 1)=x_tmp;
        end

        % generate the measurement matrix
        % 
        % 
        Phi=randn(m,n);
         % Phi = randn(m,n) + sqrt(-1)*randn(m,n);
         A=Phi./(ones(m,1)*sqrt(sum(Phi.^2)));
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

        % %% Revoery via PC-SBL
        % eta=1;
        % sigma2 =1;
        % x_csbl=MPCSBL_alternative(y,A,sigma2,eta);
        % nmse_csbl(MC)=norm(mean(abs(x_csbl),2)-mean(abs(X),2))^2/org_x;
        % 
        % % x_new=mu_new;
        % % subplot(1,5,2)
        % % nmse=norm(mean(abs(x_new),2)-mean(abs(X),2))^2/org_x
        % %
        % % stem(mean(abs(X),2),'b')
        % % hold on
        % % stem(mean(abs(x_new),2),'r')
        % % title(['EM PCSBL NMSE v1: ', num2str(nmse)])
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


        % %% Revoery via PC-SBL
        % eta=1;
        % sigma2 =1;
        % x_pcsbl=MPCSBL(y,A,sigma2,eta);
        % % x_new=mu_new;
        % % subplot(1,6,2)
        % nmse_pcsbl(MC)=norm(mean(abs(x_pcsbl),2)-mean(abs(X),2))^2/org_x;
        % 
        % % stem(mean(abs(X),2),'b')
        % % hold on
        % % stem(mean(abs(x_new),2),'r')
        % % title(['EM PCSBL NMSE: ', num2str(nmse)])


        eta=1;
        sigma2 =1;
        x_sblcorr=MSBL_correlated(y,A,sigma2,eta);
        % x_new=mu_new;
        % subplot(1,5,5)
        nmse_corr(MC)=norm(mean(abs(x_sblcorr),2)-mean(abs(X),2))^2/org_x;

        eta=1;
        sigma2 =1;
        x_sblcorrgd=MSBL_correlated_enhanced(y,A,sigma2,eta,iter_in);
        % x_new=mu_new;
        % subplot(1,5,5)
        nmse_corrgd(MC)=norm(mean(abs(x_sblcorrgd),2)-mean(abs(X),2))^2/org_x;

        eta=1;
        sigma2 =1;
        x_sblcorrfp=MSBL_correlated_enhanced_fp(y,A,sigma2,eta,iter_in);
        % x_new=mu_new;
        % subplot(1,5,5)
        nmse_corrfp(MC)=norm(mean(abs(x_sblcorrfp),2)-mean(abs(X),2))^2/org_x;

             eta=1;
        sigma2 =1;
        x_sblcorrfp2=MSBL_correlated_enhanced_fpv2(y,A,sigma2,eta,iter_in);
        % x_new=mu_new;
        % subplot(1,5,5)
        nmse_corrfp2(MC)=norm(mean(abs(x_sblcorrfp2),2)-mean(abs(X),2))^2/org_x;
       

        sup_X = zeros(size(X));
        sup_x_sbl = zeros(size(X));
        sup_x_corr = zeros(size(X));
        sup_x_corrgd = zeros(size(X));
        sup_x_corrfp = zeros(size(X));
        sup_x_corrfp2 = zeros(size(X));


        sup_X(abs(X)>rat) = 1;
        sup_x_sbl(abs(x_sbl) > rat)=1;
        sup_x_corr(abs(x_sblcorr) > rat)=1;
        sup_x_corrgd(abs(x_sblcorrgd) > rat)=1;
        sup_x_corrfp(abs(x_sblcorrfp) > rat)=1;
        sup_x_corrfp2(abs(x_sblcorrfp2) > rat)=1;

        sum_temp = zeros(size(X));
        sum_temp((sup_X+sup_x_sbl)==2) =1;
        srate_sbl(MC) = sum(sum_temp)/(sum(abs(sup_x_sbl-sup_X)) + sum(abs(sup_X))); 

      
        sum_temp = zeros(size(X));
        sum_temp((sup_X+sup_x_corr)==2) =1;
        srate_corr(MC)= sum(sum_temp)/(sum(abs(sup_x_corr-sup_X)) + sum(abs(sup_X))); 

        sum_temp = zeros(size(X));
        sum_temp((sup_X+sup_x_corr)==2) =1;
        srate_corrgd(MC)= sum(sum_temp)/(sum(abs(sup_x_corr-sup_X)) + sum(abs(sup_X))); 

        sum_temp = zeros(size(X));
        sum_temp((sup_X+sup_x_corr)==2) =1;
        srate_corrfp(MC)= sum(sum_temp)/(sum(abs(sup_x_corr-sup_X)) + sum(abs(sup_X))); 

        sum_temp = zeros(size(X));
        sum_temp((sup_X+sup_x_corr)==2) =1;
        srate_corrfp2(MC)= sum(sum_temp)/(sum(abs(sup_x_corr-sup_X)) + sum(abs(sup_X))); 

    end

   
    NMSE_set_sbl(jj,:) = nmse_sbl;
    NMSE_set_corr(jj,:) = nmse_corr;
    NMSE_set_corrgd(jj,:) = nmse_corrgd;
    NMSE_set_corrfp(jj,:) = nmse_corrfp;
    NMSE_set_corrfp2(jj,:) = nmse_corrfp2;


    sup_set_sbl(jj,:) = srate_sbl;
    sup_set_corr(jj,:) = srate_corr;
    sup_set_corrgd(jj,:) = srate_corrgd;
    sup_set_corrfp(jj,:) = srate_corrfp;
    sup_set_corrfp2(jj,:) = srate_corrfp2;

  
    jj = jj + 1;
end

%%

sup_set_sbl(sup_set_sbl<1)=0;
sup_set_corr(sup_set_corr<1)=0;
sup_set_corrgd(sup_set_corrgd<1)=0;
sup_set_corrfp(sup_set_corrfp<1)=0;
sup_set_corrfp2(sup_set_corrfp2<1)=0;




data = set;
ll=2;
plot(data,(mean(sup_set_sbl,2)),'LineWidth',ll)
hold on
plot(data,(mean(sup_set_corr,2)),'--*','LineWidth',ll)
hold on 
plot(data,(mean(sup_set_corrgd,2)),'--*','LineWidth',ll)
hold on 
plot(data,(mean(sup_set_corrfp,2)),'--*','LineWidth',ll)
hold on 
plot(data,(mean(sup_set_corrfp2,2)),'--*','LineWidth',ll)
hold on 

ylabel('Support Recovery Rate')
xlabel('K')
legend('SBL,''Proposed', 'Proposed GD', 'Proposed FP','Proposed FP2' )
grid on
title(['Correlated sources: m = ', num2str(m),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])


figure
data = set;
ll=2;
plot(data,(mean(NMSE_set_sbl,2)),'LineWidth',ll)
hold on
% hold on
plot(data,(mean(NMSE_set_corr,2)),'--*','LineWidth',ll)
hold on 
plot(data,(mean(NMSE_set_corrgd,2)),'--*','LineWidth',ll)
hold on 
plot(data,(mean(NMSE_set_corrfp,2)),'--*','LineWidth',ll)
hold on 
plot(data,(mean(NMSE_set_corrfp2,2)),'--*','LineWidth',ll)

ylabel('NMSE')
xlabel('K')
legend('SBL','Proposed', 'Proposed GD', 'Proposed FP','Proposed FP2' )
grid on
title(['Correlated sources: m = ', num2str(m),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])


NMSE_sr_sbl =zeros(size(NMSE_set_sbl));
NMSE_sr_corr =zeros(size(NMSE_set_corr));
NMSE_sr_corrgd =zeros(size(NMSE_set_corrgd));
NMSE_sr_corrfp =zeros(size(NMSE_set_corrfp));
NMSE_sr_corrfp2 =zeros(size(NMSE_set_corrfp2));

figure
k=1;

NMSE_sr_sbl((NMSE_set_sbl<10^(-k)))=1;
NMSE_sr_corr((NMSE_set_corr<10^(-k)))=1;
NMSE_sr_corrgd((NMSE_set_corrgd<10^(-k)))=1;
NMSE_sr_corrfp((NMSE_set_corrfp<10^(-k)))=1;
NMSE_sr_corrfp2((NMSE_set_corrfp2<10^(-k)))=1;




plot(data,mean(NMSE_sr_sbl,2),'LineWidth',ll)
hold on
plot(data,mean(NMSE_sr_corr,2),'--*','LineWidth',ll)
hold on 
plot(data,mean(NMSE_sr_corrgd,2),'--*','LineWidth',ll)
hold on 
plot(data,mean(NMSE_sr_corrfp,2),'--*','LineWidth',ll)
hold on 
plot(data,mean(NMSE_sr_corrfp2,2),'--*','LineWidth',ll)


ylabel('Success Rate')
xlabel('K')
legend('SBL','Proposed', 'Proposed GD', 'Proposed FP','Proposed FP2' )
grid on
title(['Correlated sources: m = ', num2str(m),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])










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

