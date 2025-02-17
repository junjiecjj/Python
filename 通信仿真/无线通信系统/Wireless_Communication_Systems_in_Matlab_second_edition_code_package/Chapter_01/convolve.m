function [y]=convolve(h,x)
%Convolve two sequences h and x of arbitrary lengths:  y=h*x
    H=convMatrix(h,length(x)); %see convMatrix.m
    y=H*x.';  %equivalent to conv(h,x) inbuilt function
end