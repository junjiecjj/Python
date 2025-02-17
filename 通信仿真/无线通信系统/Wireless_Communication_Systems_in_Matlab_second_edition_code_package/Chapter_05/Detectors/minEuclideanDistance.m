function [idealPoints,indices]= minEuclideanDistance(x,y)
%function to compute the pairwise minimum Distance between two 
%vectors x and y in p-dimensional signal space and select the 
%vectors in y that provides the minimum distances.
% x - a matrix of size mxp
% y - a matrix of size nxp. This acts as a reference against 
% which each point in x is compared.
% idealPoints - contain the decoded vector
% indices - indices of the ideal points in reference matrix y
[m,p1] = size(x);[n,p2] = size(y);

if p1~=p2
   error('Dimension Mismatch: x and y must have same dimension')
end

X = sum(x.*x,2);
Y = sum(y.*y,2)';
d = X(:,ones(1,n)) + Y(ones(1,m),:) - 2*x*y';%Squared Euclidean Dist.
[~,indices]=min(d,[],2); %Find the minimum value along DIM=2
idealPoints=y(indices,:);
indices=indices.';
end
