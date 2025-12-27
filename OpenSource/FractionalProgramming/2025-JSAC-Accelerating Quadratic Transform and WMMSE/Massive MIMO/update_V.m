function [V] = update_V(eta,w,t,chn,Y)
% t: auxiliary variable SINR (NumUser NumCell)
% Y: auxiliary variable (Rx NumUser NumCell)
% L: NumCell
% Q: NumUser in each cell
% chn: channel (Rx Tx 1 (NumUser x NumCell) NumCell)
% w: weight of each rate (NumUser NumCell)
% V: precoding vectors (Tx NumUser NumCell)
[~,M,~,L] = size(chn); % N:Rx M:TX
Q = size(t,1); V = zeros(M,Q,L);
for l = 1:L
    for q = 1:Q
        deno = eta(l)*eye(M);
        for i = 1:L
            for j = 1:Q
                X = chn(:,:,(i-1)*3+j,l)'*Y(:,j,i);
                deno = deno+w(j,i)*(1+t(j,i))*(X*X');
            end
        end
        V(:,q,l) = (deno)\(w(q,l)*(1+t(q,l))*chn(:,:,(l-1)*3+q,l)'*Y(:,q,l));
    end
end
end