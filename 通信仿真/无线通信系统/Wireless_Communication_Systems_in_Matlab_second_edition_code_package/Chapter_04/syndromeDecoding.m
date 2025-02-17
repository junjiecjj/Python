G =[0 1 1 1 1 0 0; 1 0 1 1 0 1 0;1 1 0 1 0 0 1]
H = gen2parmat(G);
[k,n] = size(G);

[stdArray] = standardArray(G);
A = stdArray;
B = H.';
%cellfun(@(x)bsxfun(@times,x,B),A,'un',0)
C = cell(size(A));
for ii = 1:numel(A)        
   C{ii} = mod(A{ii}*B,2);        
end