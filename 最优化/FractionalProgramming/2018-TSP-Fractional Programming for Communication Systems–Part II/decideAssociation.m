function [ association ] = decideAssociation( L, K, G )

association = NaN(K,1);

% max-sig association
for i = 1:K
    [~,association(i)] = max(G(i,:));
end

% %%
% for j = 1:L
%     association((1:K/L)+(j-1)*K/L) = j;
% end

end