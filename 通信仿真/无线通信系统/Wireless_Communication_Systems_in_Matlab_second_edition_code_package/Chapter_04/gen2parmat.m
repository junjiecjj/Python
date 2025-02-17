function [H] = gen2parmat(G)
%Convert standard-form binary generator matrix G
%into the corresponding parity-check matrix H.

[k,n] = size(G);%G is k x n
if k>n, error('G matrix must be of size k x n where k<n'); end
I = eye(k); %identity matrix of size k x k

%Find if the G matrix is [P | Ik] or [Ik | P] form, 
%then form H matrix accordingly
P = G(:,1:n-k); Ik=G(:,n-k+1:n); %Split G like [P | Ik] and test

if isequal(Ik,I) %G matrix is of form [P | Ik]
    H = [eye(n-k) P.'];
else
    Ik=G(:,1:k); P = G(:,k+1:n); %Split G like [Ik | P] and test
    if isequal(Ik,I) %G matrix is of form [Ik | P]
        H = [P.' eye(n-k)];
    else error('G matrix is not in standard form');
    end
end; end