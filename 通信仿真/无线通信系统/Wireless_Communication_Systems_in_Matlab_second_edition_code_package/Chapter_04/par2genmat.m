function [G] = par2genmat(H)
%Convert standard-form binary parity-check matrix H
%into the corresponding generator matrix G.
      
[r,n] = size(H); %r=n-k : num of parity bits, H matrix is (n-k) x n
k = n-r;
if r>n, error('H matrix must be of size (n-k)xn where (n-k)<n'); end
I = eye(n-k); %identity matrix of size (n-k) x (n-k)

%Find if the H matrix is [P' | I(n-k)] or [I(n-k) | P'] form, 
%then form G matrix accordingly
Pd = H(:,1:k); Ink=H(:,k+1:n); %Split H like [P' | I(n-k)] and test

if isequal(Ink,I) %H matrix is of form [P' | I(n-k)]
    G = [eye(k) Pd.'];
else
    Ik=H(:,1:n-k); Pd = H(:,n-k+1:n); %Split H like [I(n-k) | P']
    if isequal(Ik,I) %H matrix is of form [Ik | P]
        G = [Pd.' eye(k)];
    else error('H matrix is not in standard form');
    end
end; end