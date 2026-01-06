function a = Permutation(n,m)
%% Input
%       n               All
%       m               Selected
%% Output
%       a               result
%% 
if nargin ~= 2
    disp('Error: number of input parameters is not 2\n')
    return
elseif  n < m
    disp('Error: n less than m\n')
    return
end
n = floor(n);
m = floor(m);
a = factorial(n)/factorial(n-m);
end