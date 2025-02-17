function [syndromeTable] = getSyndromeTable(G)
%function to construct the syndrome table for a given standard-
%form generator matrix G. Returns a sorted cell array whose 
%first column is the syndromes (s=r.H^T) in decimal and the 
%second column is the estimated error pattern (e').

%G =[0 1 1 1 1 0 0; 1 0 1 1 0 1 0;1 1 0 1 0 0 1];%to test
H = gen2parmat(G);%Corresponding parity-check matrix

[stdArray] = standardArray(G);%construct standard array
A = stdArray; B = H.';

%multiply each element in the standard array with H^T for syndromes
S_full = cell(size(A));%for syndromes of all patterns in std array
for ii = 1:numel(A)        
   S_full{ii} = mod(A{ii}*B,2);        
end

%celldisp(S_full,'syndrome Tbl');%display syndrome for all patterns
%syndromeTable is constructed by extracting the syndromes 
%(first column) from S and the %coset leaders (first columns) from A
%Syndrome Table is sorted according to the syndrome column

S_dec=bi2de(cell2mat(S_full(:,1)),'left-msb');%syndrome in decimal
syndromeTable = [num2cell(S_dec) A(:,1)];%unsorted table
%celldisp(syndromeTable,'unsorted syndrome table'); %to display
syndromeTable = sortrows(syndromeTable,1);%sorted table
%celldisp(syndromeTable,'sorted syndrome table'); %to display