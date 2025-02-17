function [w,err,optDelay]=zf_equalizer(h,N,delay)
% Delay optimized zero forcing equalizer.
%   [w,err,optDelay]=zf_equalizer(h,N,delay) designs a ZERO FORCING equalizer
%   w for given channel impulse response h, the desired length of the equalizer
%   N and equalizer delay (delay). Also returns the equalizer error (err),the
%   best optimal delay (optDelay) that could work best for the designed equalizer
%
%   [w,err,optDelay]=zf_equalizer(h,N) designs a DELAY OPTIMIZED ZERO FORCING
%   equalizer w for given channel impulse response h, the desired length of the 
%   equalizer N. Also returns equalizer error(err),the best optimal delay(optDelay)

    h=h(:); %Channel matrix to solve simultaneous equations
    L=length(h); %length of CIR    
    H=convMatrix(h,N); %(L+N-1)xN matrix - see Chapter 1
    %section title - Methods to compute convolution
    
    %compute optimum delay based on MSE
    Hp = inv(H'*H)*H'; %Moore-Penrose Pseudo inverse
    [~,optDelay] = max(diag(H*Hp)); 
    optDelay=optDelay-1;%since Matlab index starts from 1
    k0=optDelay;
    
    if nargin==3,
        if delay >=(L+N-1), error('Too large delay'); end
        k0=delay;
    end
    d=zeros(N+L-1,1);
    d(k0+1)=1; %optimized position of equalizer delay    
    w=Hp*d;%Least Squares solution
    err=1-H(k0+1,:)*w; %equalizer error (MSE)-reference [5]    
    MSE=(1-d'*H*Hp*d);%MSE and err are equivalent
end