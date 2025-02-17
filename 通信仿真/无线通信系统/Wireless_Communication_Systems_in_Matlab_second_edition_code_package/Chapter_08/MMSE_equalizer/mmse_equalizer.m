function [w,mse,optDelay]=mmse_equalizer(h,snr,N,delay)
% Delay optimized MMSE Equalizer.
%   [w,mse]=mmse_equalizer(h,snr,N,delay) designs a MMSE equalizer w
%   for given channel impulse response h, input 'snr' (dB), the desired length
%   of the equalizer N and equalizer delay (delay). It also returns the mean
%   square error (mse) and the optimal delay (optDelay) of the designed equalizer.
%
%   [w,mse,optDelay]=mmse_equalizer(h,snr,N) designs a DELAY OPTIMIZED MMSE
%   equalizer w for given channel impulse response h, input 'snr' (dB) and
%   the desired length of the equalizer N. Also returns the mean square error(mse)
%   and the optimal delay (optDelay) of the designed equalizer.

 h=h(:);%channel matrix to solve for simultaneous equations
 L=length(h); %length of CIR    
 H=convMatrix(h,N); %(L+N-1)xN matrix - see Chapter 1
 %section title - Methods to compute convolution
 
 gamma = 10^(-snr/10); %inverse of SNR
 %compute optimum delay
 [~,optDelay] = max(diag(H*inv(H'*H+gamma*eye(N))*H')); 
 optDelay=optDelay-1; %since Matlab index starts from 1
 k0=optDelay;
 
 if nargin==4,
   if delay >=(L+N-1), error('Too large delay'); end
   k0=delay;
 end    
 d=zeros(N+L-1,1);
 d(k0+1)=1; %optimized position of equalizer delay    
 w=inv(H'*H+gamma*eye(N))*H'*d; %Least Squares solution
 mse=(1-d'*H*inv(H'*H+gamma*eye(N))*H'*d);%assume var(a)=1
end