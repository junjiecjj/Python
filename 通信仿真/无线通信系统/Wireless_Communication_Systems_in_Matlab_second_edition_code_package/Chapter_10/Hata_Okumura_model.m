%Matlab code to simulate Hata-Okumura Models
clearvars;clc;
%----------Input Section---------------
hb=70; hm=1.5; %effective BS and MS heights in meters
fc = 1500; %Frequency given in MHz
d = 1:1:100; %distances in Kms

PL = Hata_model(fc,d,hb,hm,'suburban'); semilogx(d,PL,'b'); hold on;
PL = Hata_model(fc,d,hb,hm,'open'); semilogx(d,PL,'r');
PL = Hata_model(fc,d,hb,hm,'metro'); semilogx(d,PL,'k');

title('Hata-Okumura Path Loss Model');
xlabel('Distance (km)');
ylabel('Propagation path Loss (dB)');grid on;