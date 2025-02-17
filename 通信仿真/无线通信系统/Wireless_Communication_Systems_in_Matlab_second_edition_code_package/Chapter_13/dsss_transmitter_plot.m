Rb=100; Rc=1000;L=32; %data rate, chip rate and oversampling factor
prbsType='GOLD'; %PRBS type is set to Gold code
G1=[1 1 1 1 0 0 0 1]; G2 = [1 0 0 1 0 0 0 1];%LFSR polynomials
X1 = [0 0 0 0 0 0 1]; X2=[ 0 0 0 0 0 0 1] ; %initial state of LFSRs
d = rand(1,2) >=0.5; %10 bits of random data
[s_t,carrier_ref,prbs_ref]=dsss_transmitter(d,prbsType,G1,G2,X1,X2,Rb,Rc,L);