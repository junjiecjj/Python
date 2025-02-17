function [Gv, n] = singleKnifeEdgeModel(h,f,d1,d2)
%Compute diffraction loss (G(v)dB) & the Fresnel zone number (n)
%blocked by the tip of obstruction, given the following inputs
% h - height of the knife-edge above the line joining the tx-rx (m)
% h is negative if the tip of the knifge edge is below the line
% f  - frequency of transmission (Hz)
% d1 - distance from transmitter to tip of the knife-edge (m)
% d2 - distance from receiver to tip of the knife-edge (m)

lambda = 3*10^8/f; %wavelength

v = h*sqrt(2/lambda*(1/d1 + 1/d2));%Fresnel-Kirchoff parameter
Gv = diffractionLoss(v);%approximate diffraction loss (G(v))

delta = h^2/2*(1/d1+1/d2);%path difference b/w LOS & diffracted rays
n = 2*delta/lambda; %Fresnel zone at the tip of obstruction