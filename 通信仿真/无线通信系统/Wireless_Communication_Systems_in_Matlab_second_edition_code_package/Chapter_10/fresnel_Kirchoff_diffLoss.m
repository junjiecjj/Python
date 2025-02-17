v=-5:1:20; %Range of Fresnel-Kirchoff diffraction parameter
Ld= diffractionLoss(v); %diffraction gain/loss (dB)
plot(v,-Ld);
title('Diffraction Gain Vs. Fresnel-Kirchoff parameter');
xlabel('Fresnel-Kirchoff parameter (v)');
ylabel('Diffraction gain - G_d(v) dB');