Pt_dBm=0; %Input transmitted power in dBm
Gt_dBi=1; %Gain of the Transmitted antenna in dBi
Gr_dBi=1; %Gain of the Receiver antenna in dBi
f=2.4e9; %Transmitted signal frequency in Hertz 
d0=1; %assume reference distance = 1m
d=100*(1:0.2:100); %Array of distances to simulate
L=1; %Other System Losses, No Loss case L=1
sigma=2;%Standard deviation of log Normal distribution (in dB)
n=2; % path loss exponent

%Log normal shadowing (with shadowing effect)
[PL_shadow,Pr_shadow] = logNormalShadowing(Pt_dBm,Gt_dBi,Gr_dBi,f,d0,d,L,sigma,n);
figure;plot(d,Pr_shadow,'b');hold on;

%Friis transmission (no shadowing effect)
[Pr_Friss,PL_Friss] = FriisModel(Pt_dBm,Gt_dBi,Gr_dBi,f,d,L,n);
plot(d,Pr_Friss,'r');grid on;
xlabel('Distance (m)'); ylabel('P_r (dBm)');
title('Log Normal Shadowing Model');legend('Log normal shadowing','Friss model');