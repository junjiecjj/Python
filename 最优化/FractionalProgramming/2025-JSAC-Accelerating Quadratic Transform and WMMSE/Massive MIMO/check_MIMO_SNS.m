function [flags,sns] = check_MIMO_SNS(V,Pmax)
% V: precoding vectors (Tx NumUser NumCell)
% Pamx: power constraint
% flags = 1 Meet power constraints =0 Violate power constraints
% sns: sum norm square
[~,Q,L] = size(V); sns = zeros(L,1);
flags = zeros(L,1);
for l = 1:L
    for q = 1:Q
        sns(l)=sns(l)+norm(V(:,q,l))^2;
    end
    if sns(l)<=Pmax
        flags(l) = 1;
    end
end
end