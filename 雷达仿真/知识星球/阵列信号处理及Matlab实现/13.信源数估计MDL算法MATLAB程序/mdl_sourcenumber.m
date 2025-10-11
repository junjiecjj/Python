clear all,
close all,

K    = 6;                        % Number of antenna 
snr  = -2;                       % SNR
theta    =[10,16,20];             % DOA
Sample   =  [10 20 30 40 60 80 100 120 150 200 300 400 600 900 1200]';  % Number of snapshots
C = 2;
Ntrial  = 200;

jj = sqrt(-1);

% ----------------------------------------------------------------------- %
Ndoa         = size(theta,2);
Nsample      = length(Sample);
pdf_MDL        = zeros(Nsample,Ndoa+2);
proposedMethod = zeros(Nsample,Ndoa+2);
Num_ref      = [0:Ndoa+1]';
for nNsample =1:Nsample
    number_dEVD     = zeros(Ndoa+2,1);
    
    SNR = ones(size(theta',1),1) * snr;
    T = Sample(nNsample);
    for nTrial = 1:Ntrial
        %==================================================================
        source_power = (10.^(SNR./10));
        source_amplitude = sqrt(source_power)*ones(1,T); % Source standard deviation.
        source_wave = sqrt(0.5)*(randn(T,Ndoa) + jj*randn(T,Ndoa));
        st = source_amplitude.*source_wave.';
        d0 = st(1,:).';
        nt = sqrt(0.5)*(randn(K,T)+jj*randn(K,T));
        A = exp(jj*pi*[0:K-1]'*sin(theta));
        xt = A*st + nt;       % received signal 
             
        %======================= MDL method ==============================
        [Ke,N]=size(xt);
        Rx = (xt*xt')./T;
        [u,s,v] = svd(Rx);
        sd = diag(s);
        a = zeros(1,K);
        for m = 0:K-1
            negv = sd(m+1:K);
            Tsph = mean(negv)/((prod(negv))^(1/(Ke-m)));
            a(m+1) = T*(K-m)*log(Tsph) + m*(2*K-m)*log(T)/2;
        end
        [y,b] = min(a);
        dEVD = b - 1;

        p_dEVD = find(dEVD - Num_ref == 0);
        number_dEVD(p_dEVD) = number_dEVD(p_dEVD) + 1;
        %----------------------------------------------
    end %for nTrial
    pdf_MDL(nNsample,1:end) = number_dEVD'/nTrial;
   
end %
%============================================
figure,
semilogx(Sample,pdf_MDL(:,Ndoa+1),'b:*')
legend('MDL')
ylabel('Probability of Detection')
xlabel('Number of Snapshots')
axis([Sample(1),Sample(length(Sample)),0,1])

