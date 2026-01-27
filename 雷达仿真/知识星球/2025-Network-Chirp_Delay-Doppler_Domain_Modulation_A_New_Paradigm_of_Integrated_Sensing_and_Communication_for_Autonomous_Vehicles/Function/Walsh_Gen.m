function q = Walsh_Gen(Nt, Nc)
%% Walsh code or Hadamard code generation
% input
%       Nt          # Tx
%       Nc          # chirp per frame
% output
%       q           Walsh code(Nt×Nc)
%%
tmp = 0;
for i = 1:log2(Nc)
    tmp = [tmp tmp;...
          tmp not(tmp)];
end
q = exp(1j*pi*tmp(:,Nc-Nt+1:end).');
% q = exp(1j*pi*tmp(:,:).');



end