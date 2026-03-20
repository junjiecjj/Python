
%% Initialization
clear;
close all
clc
set=[25 30 35 40 45 50 55 60];
iter = 100;
SNR = 20;    % Signal-to-noise ratio

NMSE_set_sbl = zeros(length(set),iter,4);
NMSE_set_csbl = zeros(length(set),iter,4);
NMSE_set_pcsbl = zeros(length(set),iter,4);
NMSE_set_bsbl2 = zeros(length(set),iter),4;
NMSE_set_bsbl4 = zeros(length(set),iter,4);
NMSE_set_ebsbl2 = zeros(length(set),iter,4);
NMSE_set_ebsbl4 = zeros(length(set),iter,4);
NMSE_set_corr = zeros(length(set),iter,4);
NMSE_set_corr2 = zeros(length(set),iter,4);

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
for m = set

    for MC = 1:iter

        n=100;                                          % signal dimension
        cp = 1;% number of measurements
        K=25;                                           % total number of nonzero coefficients
        L=5;                                            % number of nonzero blocks
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
        g_cum=cumsum(g);
        r = [5 5 5 5 5];
        for i=1:L
            if i == 1 || i == 3

                seg = 1/sqrt(2)*(randn(r(i), 1) +1j*randn(r(i),1));              % generate the non-zero bloc
                seg = 0*seg;
                cp = 0;
                cp2 = 0;
                seg(3) =1/sqrt(2)*(randn +1j*randn);
                R = eye(r(i)) + diag(cp*ones(r(i)-1,1),1)+diag(cp*ones(r(i)-1,1),-1)+diag(cp2*ones(r(i)-2,1),2)+diag(cp2*ones(r(i)-2,1),-2);
                seg=sqrtm(R)*seg;
                loc=randperm(g(i)-r(i));        % the starting position of non-zero block
                x_tmp=zeros(g(i), 1);
                x_tmp(loc(1):loc(1)-1+r(i))= seg;
                x(g_cum(i)-g(i)+1:g_cum(i), 1)=x_tmp;

            elseif i == 4


                seg = 1/sqrt(2)*(randn(r(i), 1) +1j*randn(r(i),1));              % generate the non-zero bloc
                cp = 0;
                cp2 = 0;
                seg(4) =0;
                R = eye(r(i)) + diag(cp*ones(r(i)-1,1),1)+diag(cp*ones(r(i)-1,1),-1)+diag(cp2*ones(r(i)-2,1),2)+diag(cp2*ones(r(i)-2,1),-2);
                seg=sqrtm(R)*seg;
                loc=randperm(g(i)-r(i));        % the starting position of non-zero block
                x_tmp=zeros(g(i), 1);
                x_tmp(loc(1):loc(1)-1+r(i))= seg;
                x(g_cum(i)-g(i)+1:g_cum(i), 1)=x_tmp;

            else
                % generate i-th block
                seg = 1/sqrt(2)*(randn(r(i), 1) +1j*randn(r(i),1));              % generate the non-zero bloc
                cp = 0;
                cp2 = 0;
                R = eye(r(i)) + diag(cp*ones(r(i)-1,1),1)+diag(cp*ones(r(i)-1,1),-1)+diag(cp2*ones(r(i)-2,1),2)+diag(cp2*ones(r(i)-2,1),-2);
                seg=sqrtm(R)*seg;
                loc=randperm(g(i)-r(i));        % the starting position of non-zero block
                x_tmp=zeros(g(i), 1);
                x_tmp(loc(1):loc(1)-1+r(i))= seg;
                x(g_cum(i)-g(i)+1:g_cum(i), 1)=x_tmp;
            end
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
        indexes = find(x~=0);
        org_x = norm(mean(abs(X),2))^2;
        i1=[1:g_cum(1) g_cum(2)+1: g_cum(3)];
        i2=[1+g_cum(3):g_cum(4)];
        i3=[1+g_cum(1):g_cum(2) g_cum(4)+1: g_cum(5)];
        org_x = norm(mean(abs(X),2))^2;
        org_x1 = norm(mean(abs(X(i1)),2))^2;
        org_x2 = norm(mean(abs(X(i2)),2))^2;
        org_x3 = norm(mean(abs(X(i3)),2))^2;

        %% Revoery via PC-SBL
        eta=1;
        sigma2 =1;
        x_csbl=MPCSBL_alternative(y,A,sigma2,eta);
        nmse_csbl(MC,1)=norm(mean(abs(x_csbl),2)-mean(abs(X),2))^2/org_x;
        nmse_csbl(MC,2)=norm(mean(abs(x_csbl(i1)),2)-mean(abs(X(i1)),2))^2/org_x1;
        nmse_csbl(MC,3)=norm(mean(abs(x_csbl(i2)),2)-mean(abs(X(i2)),2))^2/org_x2;
        nmse_csbl(MC,4)=norm(mean(abs(x_csbl(i3)),2)-mean(abs(X(i3)),2))^2/org_x3;
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
        nmse_sbl(MC,1)=norm(mean(abs(x_sbl),2)-mean(abs(X),2))^2/org_x;
        nmse_sbl(MC,2)=norm(mean(abs(x_sbl(i1)),2)-mean(abs(X(i1)),2))^2/org_x1;
        nmse_sbl(MC,3)=norm(mean(abs(x_sbl(i2)),2)-mean(abs(X(i2)),2))^2/org_x2;
        nmse_sbl(MC,4)=norm(mean(abs(x_sbl(i3)),2)-mean(abs(X(i3)),2))^2/org_x3;
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
        nmse_pcsbl(MC,1)=norm(mean(abs(x_pcsbl),2)-mean(abs(X),2))^2/org_x;
        nmse_pcsbl(MC,2)=norm(mean(abs(x_pcsbl(i1)),2)-mean(abs(X(i1)),2))^2/org_x1;
        nmse_pcsbl(MC,3)=norm(mean(abs(x_pcsbl(i2)),2)-mean(abs(X(i2)),2))^2/org_x2;
        nmse_pcsbl(MC,4)=norm(mean(abs(x_pcsbl(i3)),2)-mean(abs(X(i3)),2))^2/org_x3;
        % stem(mean(abs(X),2),'b')
        % hold on
        % stem(mean(abs(x_new),2),'r')
        % title(['EM PCSBL NMSE: ', num2str(nmse)])


        eta=0.5;
        sigma2 =1;
        x_sblcorr=MSBL_correlated(y,A,sigma2,eta);
        % x_new=mu_new;
        % subplot(1,5,5)
        nmse_corr(MC,1)=norm(mean(abs(x_sblcorr),2)-mean(abs(X),2))^2/org_x;
        nmse_corr(MC,2)=norm(mean(abs(x_sblcorr(i1)),2)-mean(abs(X(i1)),2))^2/org_x1;
        nmse_corr(MC,3)=norm(mean(abs(x_sblcorr(i2)),2)-mean(abs(X(i2)),2))^2/org_x2;
        nmse_corr(MC,4)=norm(mean(abs(x_sblcorr(i3)),2)-mean(abs(X(i3)),2))^2/org_x3;

        eta=0.5;
        sigma2 =1;
        x_sblcorr2=MSBL_correlated_second(y,A,sigma2,eta);
        % x_new=mu_new;
        % subplot(1,5,5)
        nmse_corr2(MC,1)=norm(mean(abs(x_sblcorr2),2)-mean(abs(X),2))^2/org_x;
        nmse_corr2(MC,2)=norm(mean(abs(x_sblcorr2(i1)),2)-mean(abs(X(i1)),2))^2/org_x1;
        nmse_corr2(MC,3)=norm(mean(abs(x_sblcorr2(i2)),2)-mean(abs(X(i2)),2))^2/org_x2;
        nmse_corr2(MC,4)=norm(mean(abs(x_sblcorr2(i3)),2)-mean(abs(X(i3)),2))^2/org_x3;        %%
        blkLen =2;

        blkStartLoc = 1:blkLen:n;
        learnLambda = 2;
        Result2 = BSBL_EM(A,y,blkStartLoc,learnLambda);
        x_bsbl2 = Result2.x;
        nmse_bsbl2(MC,1)=norm(mean(abs(x_bsbl2),2)-mean(abs(X),2))^2/org_x;
        nmse_bsbl2(MC,2)=norm(mean(abs(x_bsbl2(i1)),2)-mean(abs(X(i1)),2))^2/org_x1;
        nmse_bsbl2(MC,3)=norm(mean(abs(x_bsbl2(i2)),2)-mean(abs(X(i2)),2))^2/org_x2;
        nmse_bsbl2(MC,4)=norm(mean(abs(x_bsbl2(i3)),2)-mean(abs(X(i3)),2))^2/org_x3;


        %%
        blkLen =4;

        blkStartLoc = 1:blkLen:n;
        learnLambda = 2;
        Result2 = BSBL_EM(A,y,blkStartLoc,learnLambda);
        x_bsbl4 = Result2.x;
        nmse_bsbl4(MC,1)=norm(mean(abs(x_bsbl4),2)-mean(abs(X),2))^2/org_x;
        nmse_bsbl4(MC,2)=norm(mean(abs(x_bsbl4(i1)),2)-mean(abs(X(i1)),2))^2/org_x1;
        nmse_bsbl4(MC,3)=norm(mean(abs(x_bsbl4(i2)),2)-mean(abs(X(i2)),2))^2/org_x2;
        nmse_bsbl4(MC,4)=norm(mean(abs(x_bsbl4(i3)),2)-mean(abs(X(i3)),2))^2/org_x3;
        %%
        blkLen =2;
        blkStartLoc = 1:blkLen:n;
        % = 2: small noise;
        % = 0 : no noise
        Result5 = EBSBL_BO(A, y, blkLen, noiseFlag5);

        x_ebsbl2 = Result5.x;
        nmse_ebsbl2(MC,1)=norm(mean(abs(x_ebsbl2),2)-mean(abs(X),2))^2/org_x;
        nmse_ebsbl2(MC,2)=norm(mean(abs(x_ebsbl2(i1)),2)-mean(abs(X(i1)),2))^2/org_x1;
        nmse_ebsbl2(MC,3)=norm(mean(abs(x_ebsbl2(i2)),2)-mean(abs(X(i2)),2))^2/org_x2;
        nmse_ebsbl2(MC,4)=norm(mean(abs(x_ebsbl2(i3)),2)-mean(abs(X(i3)),2))^2/org_x3;        %%
        blkLen =4;
        blkStartLoc = 1:blkLen:n;
        % = 1: strong noise;
        % = 2: small noise;
        % = 0 : no noise
        Result4 = EBSBL_BO(A, y, blkLen, noiseFlag5);

        x_ebsbl4 = Result4.x;
        nmse_ebsbl4(MC,1)=norm(mean(abs(x_ebsbl4),2)-mean(abs(X),2))^2/org_x;
        nmse_ebsbl4(MC,2)=norm(mean(abs(x_ebsbl4(i1)),2)-mean(abs(X(i1)),2))^2/org_x1;
        nmse_ebsbl4(MC,3)=norm(mean(abs(x_ebsbl4(i2)),2)-mean(abs(X(i2)),2))^2/org_x2;
        nmse_ebsbl4(MC,4)=norm(mean(abs(x_ebsbl4(i3)),2)-mean(abs(X(i3)),2))^2/org_x3;

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
        sup_x_bsbl2(abs(x_bsbl2) > rat)=1;
        sup_x_bsbl4(abs(x_bsbl4) > rat)=1;
        sup_x_ebsbl2(abs(x_ebsbl2) > rat)=1;
        sup_x_ebsbl4(abs(x_ebsbl4) > rat)=1;
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

        sum_temp = zeros(size(X));
        sum_temp((sup_X+sup_x_bsbl2)==2) =1;
        srate_bsbl2(MC) = sum(sum_temp)/(sum(abs(sup_x_bsbl2-sup_X)) + sum(abs(sup_X)));

        sum_temp = zeros(size(X));
        sum_temp((sup_X+sup_x_bsbl4)==2) =1;
        srate_bsbl4(MC)= sum(sum_temp)/(sum(abs(sup_x_bsbl4-sup_X)) + sum(abs(sup_X)));

        sum_temp = zeros(size(X));
        sum_temp((sup_X+sup_x_ebsbl2)==2) =1;
        srate_ebsbl2(MC) = sum(sum_temp)/(sum(abs(sup_x_ebsbl2-sup_X)) + sum(abs(sup_X)));

        sum_temp = zeros(size(X));
        sum_temp((sup_X+sup_x_ebsbl4)==2) =1;
        srate_ebsbl4(MC) = sum(sum_temp)/(sum(abs(sup_x_ebsbl4-sup_X)) + sum(abs(sup_X)));

    end

    % nmse4(nmse4>nmse3) = 0;
    for index = 1:4
        NMSE_set_csbl(jj,:,index) = nmse_csbl(:,index) ;
        NMSE_set_sbl(jj,:,index) = nmse_sbl(:,index);
        NMSE_set_pcsbl(jj,:,index) = nmse_pcsbl(:,index);
        NMSE_set_corr(jj,:,index) = nmse_corr(:,index);
        NMSE_set_corr2(jj,:,index) = nmse_corr2(:,index);
        NMSE_set_bsbl2(jj,:,index) = nmse_bsbl2(:,index);
        NMSE_set_bsbl4(jj,:,index) = nmse_bsbl4(:,index);
        NMSE_set_ebsbl2(jj,:,index) = nmse_ebsbl2(:,index);
        NMSE_set_ebsbl4(jj,:,index) = nmse_ebsbl4(:,index);
    end

    % nmse4(nmse4>nmse3) = 0;
    sup_set_csbl(jj,:) = srate_csbl;
    sup_set_sbl(jj,:) = srate_sbl;

    sup_set_pcsbl(jj,:) = srate_pcsbl;
    sup_set_corr(jj,:) = srate_corr;
    sup_set_bsbl2(jj,:) = srate_bsbl2;
    sup_set_bsbl4(jj,:) = srate_bsbl4;
    sup_set_ebsbl2(jj,:) = srate_ebsbl2;
    sup_set_ebsbl4(jj,:) = srate_ebsbl4;
    jj = jj + 1;
end

%%
close all
% set = set/100;
sup_set_sbl(sup_set_sbl<1)=0;
sup_set_csbl(sup_set_csbl<1)=0;
sup_set_pcsbl(sup_set_pcsbl<1)=0;
sup_set_bsbl2(sup_set_bsbl2<1)=0;
sup_set_bsbl4(sup_set_bsbl4<1)=0;
sup_set_ebsbl2(sup_set_ebsbl2<1)=0;
sup_set_ebsbl4(sup_set_ebsbl4<1)=0;
sup_set_corr(sup_set_corr<1)=0;



%
% data = set;
% ll=2;
% plot(data,(mean(sup_set_sbl,2)),'LineWidth',ll)
% hold on
% plot(data,(mean(sup_set_csbl,2)),'LineWidth',ll)
% hold on
% plot(data,(mean(sup_set_pcsbl,2)),'LineWidth',ll)
% hold on
% plot(data,(mean(sup_set_bsbl2,2)),'LineWidth',ll)
% hold on
% plot(data,(mean(sup_set_bsbl4,2)),'LineWidth',ll)
% hold on
% plot(data,(mean(sup_set_ebsbl2,2)),'LineWidth',ll)
% hold on
% plot(data,(mean(sup_set_ebsbl4,2)),'LineWidth',ll)
% hold on
% plot(data,(mean(sup_set_corr,2)),'--*','LineWidth',ll)
%
% ylabel('Support Recovery Rate')
% xlabel('m/n')
% legend('SBL', 'CSBL','PCSBL', 'BSBL(h=2)','BSBL(h=4)','EBSBL(h=2)','EBSBL(h=4)','Proposed' )
% grid on
% title(['Uncorrelated sources: K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])
%

data = set;
ll=2;
for index =1:4
    figure
    plot(data,(mean(NMSE_set_sbl(:,:,index),2)),'LineWidth',ll)
    hold on
    plot(data,(mean(NMSE_set_csbl(:,:,index),2)),'LineWidth',ll)
    hold on
    plot(data,(mean(NMSE_set_pcsbl(:,:,index),2)),'LineWidth',ll)
    hold on
    plot(data,(mean(NMSE_set_bsbl2(:,:,index),2)),'LineWidth',ll)
    hold on
    plot(data,(mean(NMSE_set_bsbl4(:,:,index),2)),'LineWidth',ll)
    hold on
    plot(data,(mean(NMSE_set_ebsbl2(:,:,index),2)),'LineWidth',ll)
    hold on
    plot(data,(mean(NMSE_set_ebsbl4(:,:,index),2)),'LineWidth',ll)
    hold on
    plot(data,(mean(NMSE_set_corr(:,:,index),2)),'--*','LineWidth',ll)
    % hold on
    % plot(data,(mean(NMSE_set_corr2,2)),'--*','LineWidth',ll)

    ylabel('NMSE')
    xlabel('m/n')
    legend('SBL', 'CSBL','PCSBL', 'BSBL(h=2)','BSBL(h=4)','EBSBL(h=2)','EBSBL(h=4)','Proposed' )
    grid on
    title(['Uncorrelated sources: K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])

    NMSE_sr_bsbl2 =zeros(size(NMSE_set_bsbl2));
    NMSE_sr_bsbl4 =zeros(size(NMSE_set_bsbl4));
    NMSE_sr_ebsbl2 =zeros(size(NMSE_set_ebsbl2));
    NMSE_sr_ebsbl4 =zeros(size(NMSE_set_ebsbl4));
    NMSE_sr_csbl =zeros(size(NMSE_set_csbl));
    NMSE_sr_sbl =zeros(size(NMSE_set_sbl));
    NMSE_sr_pcsbl =zeros(size(NMSE_set_pcsbl));
    NMSE_sr_corr =zeros(size(NMSE_set_corr));
    NMSE_sr_corr2 =zeros(size(NMSE_set_corr2));

    % figure
    % k=3;
    %
    % NMSE_sr_bsbl2((NMSE_set_bsbl2<10^(-k)))=1;
    % NMSE_sr_bsbl4((NMSE_set_bsbl4<10^(-k)))=1;
    % NMSE_sr_ebsbl2((NMSE_set_ebsbl2<10^(-k)))=1;
    % NMSE_sr_ebsbl4((NMSE_set_ebsbl4<10^(-k)))=1;
    % NMSE_sr_csbl((NMSE_set_csbl<10^(-k)))=1;
    % NMSE_sr_sbl((NMSE_set_sbl<10^(-k)))=1;
    % NMSE_sr_pcsbl((NMSE_set_pcsbl<10^(-k)))=1;
    % NMSE_sr_corr((NMSE_set_corr<10^(-k)))=1;
    % NMSE_sr_corr2((NMSE_set_corr2<10^(-k)))=1;
    %
    %
    % plot(data,mean(NMSE_sr_sbl(:,:,index),2),'LineWidth',ll)
    % hold on
    % plot(data,mean(NMSE_sr_csbl(:,:,index),2),'LineWidth',ll)
    % hold on
    % plot(data,mean(NMSE_sr_pcsbl(:,:,index),2),'LineWidth',ll)
    % hold on
    % plot(data,mean(NMSE_sr_bsbl2(:,:,index),2),'LineWidth',ll)
    % hold on
    % plot(data,mean(NMSE_sr_bsbl4(:,:,index),2),'LineWidth',ll)
    % hold on
    % plot(data,mean(NMSE_sr_ebsbl2(:,:,index),2),'LineWidth',ll)
    % hold on
    % plot(data,mean(NMSE_sr_ebsbl4(:,:,index),2),'LineWidth',ll)
    % hold on
    % plot(data,mean(NMSE_sr_corr(:,:,index),2),'--*','LineWidth',ll)
    % % hold on
    % % plot(data,mean(NMSE_sr_corr2,2),'--*','LineWidth',ll)
    %
    % ylabel('Success Rate')
    % xlabel('m/n')
    % legend('SBL', 'CSBL','PCSBL', 'BSBL(h=2)','BSBL(h=4)','EBSBL(h=2)','EBSBL(h=4)','Proposed' )
    % grid on
    % title(['Uncorrelated sources: K = ', num2str(K),  ', L = ', num2str(L), ', SNR = ', num2str(SNR)])
end











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

