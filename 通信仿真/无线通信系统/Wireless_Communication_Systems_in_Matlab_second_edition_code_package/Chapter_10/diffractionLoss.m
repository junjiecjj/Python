function [Gv]= diffractionLoss(v)
%Compute diffraction loss G(v)dB for Fresnel-Kirchoff parameter(v)
%according to Rec. ITU-R P.526-5
Gv = zeros(1,length(v));
idx = (v>-0.7); %indices where v>0.7
Gv(idx) = 6.9 + 20*log10(sqrt((v(idx)-0.1).^2 + 1) + v(idx) -0.1);
Gv(~idx)=0;%for v<-0.7
end