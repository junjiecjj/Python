function [b,t]=PRSignaling(Q,L,Nsym)
%Generate the impulse response of a partial response System given,
%Q - partial response polynomial for Q(D)
%L - oversampling factor (Tsym/Ts)
%Nsym - filter span in symbol durations
%Returns the impulse response b(t) that spans the 
%discrete-time base t=-Nsym:1/L:Nsym

%excite the Q(f) filter with an impulse to get the PR response
qn =  filter(Q,1,[ 0 0 0 0 0 1 0 0 0 0 0 0]);
%Partial response filter Q(D) <-> q(t) and its upsampled version
q=[qn ;zeros(L-1,length(qn))];%Insert L-1 zero between each symbols
q=q(:).';%Convert to a single stream, output is at sampling rate
%convolve q(t) with Nyqyist criterion satisfying sinc filter g(t) 
%Note: any filter that satisfy Nyquist first criterion can be used
Tsym=1; %g(t) generated for 1 symbol duration
t=-(Nsym/2):1/L:(Nsym/2);%discrete-time base for 1 symbol duration
g = sin(pi*t/Tsym)./(pi*t/Tsym); g(isnan(g)==1)=1; %sinc function
b = conv(g,q,'same');%convolve q(t) and g(t)
end