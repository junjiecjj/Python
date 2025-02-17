function [stdArray] = standardArray(G)
%Construct standard array for a given Generator matrix G
%G = [0 1 1 1 1 0 0; %Sample Generator matrix to test
%     1 0 1 1 0 1 0;
%     1 1 0 1 0 0 1];
[k,n] = size(G) ; %k and n derived from G matrix
M = dec2bin(0:2^k-1)-'0'; %all possible information sequences
C = mod(M*G,2);% all possible codewords - codebook
%Initialize standard array using codewords as the top row
S = C; stdArray = mat2cell(C,ones(1,2^k),n).'; 
for d=1:n,%loop for all possible error patterns (coset leader)
 [E,nRows,~] = error_pattern_combos(d,n);%all d-bit error patterns
 for j=1:nRows,%For each error pattern combination
  if sum(ismember(E(j,:),S,'rows'))==0,
   %the pattern should not exist already in S
   coset = mod(C + repmat(E(j,:),2^k,1),2);%Codeword+error pattern
   S = vertcat(S,coset);%temp matrix
   stdArray = vertcat(stdArray,mat2cell(coset,ones(1,2^k),n).');
  end
 end; end


