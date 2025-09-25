function [ instRate, sinr ] = computeInstRate( obj, schedule, V )

L = obj.numBS;
K = obj.numUser;
M = obj.numTxAnte;
N = obj.numRxAnte;
T = obj.numTone;
H = obj.chn;
noise = obj.noise;
BW = obj.bandwidth;

instRate = zeros(K,1);
sinr = nan(T,L,M);

for j = 1:L
    for s = 1:M
        for z = 1:T
            i = schedule(z,j,s);
            B = eye(N)*noise;
            for n = 1:L
                for t = 1:M
                    if n==j && t==s
                        continue
                    end
                    B = B + H(:,:,z,i,n)*V(:,z,n,t)*V(:,z,n,t)'*H(:,:,z,i,n)';
                end
            end
            sinr(z,j,s) =  abs(V(:,z,j,s)'*H(:,:,z,i,j)'*( B\(H(:,:,z,i,j)*V(:,z,j,s)) ));
            instRate(i) = instRate(i) + BW/T*log2(1+sinr(z,j,s));
        end
    end
end      

end

