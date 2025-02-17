%m-sequence autocorrelation
y=lfsr([1 0 0 1 0 1],[0 0 0 0 1]) %g(x)=x^5+x^2+1
N=31;%Period of the sequence N=2^L-1
Ryy=1/N*sequence_correlation(y,y,-35,35)%normalized autocorr.
plot(-35:1:35,Ryy) %Plot autocorrelation

%m-sequence cross-correlation
x=lfsr([1 1 0 1 1 1],[0 0 0 0 1]) %g_1(x)=x^5+x^4+x^2+x+1
y=lfsr([1 1 1 0 1 1],[0 0 0 0 1]) %g_2(x)=x^5+x^4+x^3+x+1
N=31; %Period of the sequence N=2^L-1
Rxy=1/N*sequence_correlation(x,y,0,31)%cross-correlation
plot(0:1:31,Rxy)%Plot cross-correlation