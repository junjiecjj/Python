%Gold codes auto-correlation
G1=[1 1 1 1 0 1]; G2=[1 0 0 1 0 1]; %feedback connections
X1=[ 0 0 0 0 1]; X2=[ 0 0 0 0 1]; %initial states of LFSRs
y1=lfsr(G1,X1); y2=lfsr(G2,X2);N=31; %m-sequence 1 and 2
Ry1y2= 1/N*sequence_correlation(y1,y2,0,31);%cross-correlation
plot(0:1:31,Ry1y2)%plot correlation

%Gold codes cross-correlation
N=31;%period of Gold code
G1=[1 1 1 1 0 1]; G2=[1 0 0 1 0 1]; %feedback connections
X1=[ 0 0 0 0 1]; X2=[ 0 0 0 0 1]; %initial states of LFSRs
y=gold_code_generator(G1,G2,X1,X2);%Generate Gold code
Ryy= 1/N*sequence_correlation(y,y,0,31);%auto-correlation
plot(0:1:31,Ryy);%plot auto-correlation