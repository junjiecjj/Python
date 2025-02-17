function [z]= iq_imbalance(r,g,phi)
%Function to create IQ imbalance impairment in a complex baseband 
%  [z]=iq_imbalance(r,g,phi) introduces IQ imbalance and phase error 
%  signal between the inphase and quadrature components of the 
%  complex baseband signal r. The model parameter g represents the 
%  gain mismatch between the IQ branches of the receiver and 'phi'   
%  represents the phase error of the local oscillator (in degrees).
Ri=real(r); Rq=imag(r);
Zi= Ri; %I branch
Zq= g*(-sin(phi/180*pi)*Ri + cos(phi/180*pi)*Rq);%Q branch crosstalk
z=Zi+1i*Zq;
end