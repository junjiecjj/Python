clear ; close all
 
M = 40; 
N = 40; 

d_lambda = 0.5;   
freqs_deg = [-22 -20  10  37];    % Actual DOAs (degree)   
SNR_dB = 2;           % SNR (dB)
QQQ = 720;  
sigma2noise = 1; % Noise variance
K = length(freqs_deg); % Number of complex exponentials (or the number of sources). 
freqs_rad = freqs_deg*pi/180;    
sigma2source = 10^(0.1*SNR_dB)*sigma2noise;  % Signal power
idxR = (0:(M-1))'; % Used for the sample covariance matrix

A = zeros(M,K);
for i = 1:K
    A(:,i) = exp(-1j*2*pi*d_lambda*sin(freqs_rad(i))*idxR);
end

s = zeros(K,N);
noise = zeros(M,N);
x0 = zeros(M,N);
x0_one = zeros(M,N);
 
R_one = zeros(M,M);   
        
for i = 1:N
        s(:,i) = sqrt(sigma2source/2)*(randn(K,1)+1j*randn(K,1)); 
        noise(:,i) = sqrt(sigma2noise/2)*(randn(M,1)+1j*randn(M,1));
        x0(:,i) = A*s(:,i) + noise(:,i);
        x0_one(:,i)= sign(real(x0(:,i))) + 1j*sign(imag(x0(:,i)));
        R_one = R_one + x0_one(:,i)*(x0_one(:,i)');
end
     
R_one = R_one/N;
    
[Q0,D0] = eig(R_one);
[~,idx] = sort(abs(diag(D0))); % Ascending order: lambdahat1 <= lambdahat2 <= ...
Q = Q0(:,idx);
G_one = Q(:,1:(M-K));
      
%% METHODS TESTED
[one_BMUSIC_DOA_degree,one_BMUSIC_DOA_radian] = music(G_one,K,M);  % One-bit MUSIC   

[OGIR_DOA_degree,OGIR_DOA_radian]= OGIR_1Bit(x0_one,361,M,N,300,10,10^(-4),10,10^(-5),K);   % OGIR

[CBIHT_DOA_degree,CBIHT_DOA_radian] = cbiht(x0_one,360,K,M);    % CBIHT

[GrSBL_DOA_degree,GrSBL_DOA_radian] = Gr_SBL_1Bit(x0_one,K,M,N,10,1,0,361);  % Gr-SBL

[OB_rootMUSIC_DOA_degree,OB_rootMUSIC_DOA_radian] = root_music_doa(G_one,M,K);  % One-bit root-MUSIC


% Proposed Methods: 
[OG_AdaBoost_DOA_degree,OG_AdaBoost_DOA_radian] = OG_AdaBoost(x0_one,K,M,N,20,QQQ);
        
[MUSIC_AdaBoost_DOA_degree,MUSIC_AdaBoost_DOA_radian] = MUSIC_AdaBoost(x0_one,K,M,N,20,K+1);

   