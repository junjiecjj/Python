function [obj,SINR] = Compute_MIMO_obj(sigma,M,N,L,Q,chn,w,V)
% sigma: noise power
% M: Tx
% N: Rx
% L: NumCell
% Q: NumUser in each cell
% chn: channel (Rx Tx (NumUser x NumCell) NumCell)
% w: weight of each rate (NumUser NumCell)
% V: precoding vectors (Tx NumUser NumCell)
SINR = zeros(Q,L);
for l = 1:L
    for q = 1:Q
        noise = sigma*eye(N);
        for i = 1:L
            for j = 1:Q
                noise = noise+chn(:,:,(l-1)*3+q,i)*V(:,j,i)*V(:,j,i)'*chn(:,:,(l-1)*3+q,i)';
            end
        end
        noise = noise-chn(:,:,(l-1)*3+q,l)*V(:,q,l)*V(:,q,l)'*chn(:,:,(l-1)*3+q,l)';
        SINR(q,l) = V(:,q,l)'*chn(:,:,(l-1)*3+q,l)'/(noise)*chn(:,:,(l-1)*3+q,l)*V(:,q,l);
    end
end
SINR = real(SINR);
obj = (sum(sum(log2(1+SINR).*w)));
end