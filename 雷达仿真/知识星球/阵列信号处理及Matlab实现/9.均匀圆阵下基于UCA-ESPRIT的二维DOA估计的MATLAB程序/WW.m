function w=WW(a,n)

for i=0:1:n-1
    w(i+1)=(1/16)*exp(j*2*pi*a*i/n);
    %w=w';
end