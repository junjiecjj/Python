function [M] = hadamard_matrix(m)
%Construct Hadamard matrix of order 2^m, m is positive integer
if m<=0, error('Input m has to be positive'); end;
M2=[1 1;1 -1];%order 2 Hadamard matrix
if m==1, M=M2;
else
    Mi=M2;
    for i=2:m, Mi=kron(M2,Mi); end
    M=Mi;%order 2^m Hadamard matrix
end; end