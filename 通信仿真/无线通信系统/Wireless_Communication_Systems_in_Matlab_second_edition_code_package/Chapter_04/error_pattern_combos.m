function [E,nRows,nCols] = error_pattern_combos(d,n)
%Returns all error patterns of length n with d-bit errors
%E is the matrix containing all the error pattern
%nRows and nCols are the size of the returned matrix
E = dec2bin(2^d-1,n)-'0';% %patterns with errors at d-bit errors
p = size(E,2);
q = sum(E==1);
C = nchoosek(1:p,q);%combinations of p things taken q at a time
m = size(C,1); 
E = zeros(m,p);%form an all zero matrix of error patterns
E(repmat((1-m:0)',1,q)+m*C) = 1;%1s corresponding to indices in C
[nRows,nCols]= size(E);%num of rows and columns E matrix