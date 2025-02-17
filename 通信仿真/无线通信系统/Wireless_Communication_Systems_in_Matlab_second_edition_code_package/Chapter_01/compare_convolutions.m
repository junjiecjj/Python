x=randn(1,7)+1i*randn(1,7) %Create random vectors for test
h=randn(1,3)+1i*randn(1,3) %Create random vectors for test

L=length(x)+length(h)-1; %length of convolution output

y1=convolve(h,x) %Convolution Using Toeplitz matrix
y2=ifft(fft(x,L).*(fft(h,L))).' %Convolution using FFT
y3=conv(h,x) %Matlab's standard function