%Matlab code to simulate Friis Free space equation
%-----------Input section------------------------
Pt_dBm=0; %Input - Transmitted power in dBm
Gt_dBi=1; %Gain of the Transmitted antenna in dBi
Gr_dBi=1; %Gain of the Receiver antenna in dBi
d =2.^[0,1,2,3,4,5] ; %Array of input distances in meters
L=1; %Other System Losses, No Loss case L=1
n=2; %Path loss exponent for Free space
f=2.4e9; %Transmitted signal frequency in Hertz (2.4G WiFi)
%----------------------------------------------------
[Pr1_dBm,PL1_dB] = FriisModel(Pt_dBm,Gt_dBi,Gr_dBi,f,d,L,n);
semilogx(d,Pr1_dBm,'b-o');hold on; 

f=5e9; %Transmitted signal frequency in Hertz (5G WiFi)
[Pr2_dBm,PL2_dB] = FriisModel(Pt_dBm,Gt_dBi,Gr_dBi,f,d,L,n);
semilogx(d,Pr2_dBm,'r-o');grid on;title('Free space path loss');
xlabel('Distance (m)');ylabel('Received power (dBm)')