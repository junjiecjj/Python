function h=wide(w,M)
a=(M-1)/2;
n=[0:1:(M-1)];
m=n-a+eps;
h=sin(w*m)./(pi*m);