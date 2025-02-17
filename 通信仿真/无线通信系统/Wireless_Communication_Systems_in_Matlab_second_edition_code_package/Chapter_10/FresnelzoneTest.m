d=25e3; %total distance between the tx and the Rx
f=12e9; %frequency of transmission
n=1;% Freznel zone number - affects r_n only
d1=25e3/2; d2=25e3/2; %measurement at mid point
%r_n = radius of the given zone number 
%r_clear = clearance required at first zone
[r_n,r_clear] = Fresnelzone(d1,d2,f,1)