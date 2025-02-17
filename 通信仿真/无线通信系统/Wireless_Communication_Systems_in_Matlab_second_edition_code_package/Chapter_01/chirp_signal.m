function g=chirp_signal(t,f0,t1,f1,phase)
%g = chirp_signal(t,f0,t1,f1) generates samples of a linearly 
%swept-frequency signal at the time instances defined in timebase 
%array t.  The instantaneous frequency at time 0 is f0 Hertz.  
%The instantaneous frequency f1 is achieved at time t1. The argument 
%'phase' is optional. It defines the initial phase of the signal 
%defined in radians. By default phase=0 radian
if nargin==4, phase=0; end
t0=t(1); T=t1-t0; k=(f1-f0)/T;
g=cos(2*pi*(k/2*t+f0).*t+phase);
end