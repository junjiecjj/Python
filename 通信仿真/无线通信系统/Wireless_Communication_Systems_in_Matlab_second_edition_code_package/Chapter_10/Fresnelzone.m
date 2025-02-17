function [r_n,r_clear] = Fresnelzone(d1,d2,f,n)
%Compute radius of the nth Fresnel zone - r_n and the required 
%clearance in first fresnel zone, given 
% d1 - distance from transmitter to the point of measument (m)
% d2 - distance from receiver to the point of measument (m)
% f  - frequency of transmission (Hz)
% n  - zone number for which the radius has to be calculated
% Returns the following params at the point of measurement 
% r_n - radius of Fresnel zone at the point of measurement (m)
% r_clear -first zone clearance required at measurement point(m)
lambda = 3*10^8/f; %wavelength
r_n= sqrt(n*lambda*d1*d2./(d1+d2));
%clearance required at 1st zone is 60% of 1st zone radius
r_clear = 0.6*sqrt(1*lambda*d1*d2./(d1+d2));