function [H]=convMatrix(h,p)
%Construct the convolution matrix of size (N+p-1)x p from the input 
%matrix h of size N.
 h=h(:).';
 col=[h zeros(1,p-1)]; row=[h(1) zeros(1,p-1)];
 H=toeplitz(col,row);
end