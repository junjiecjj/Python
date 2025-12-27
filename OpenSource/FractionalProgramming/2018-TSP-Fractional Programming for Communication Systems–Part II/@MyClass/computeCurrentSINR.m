function [ currentSINR ] = computeCurrentSINR(obj, schedule, V)

L = obj.numBS;
K = obj.numUser;
N = obj.numRxAnte;
H = obj.chn;
noise = obj.noise;
currentSINR = zeros(K,1);

for j = 1:L
    for s = 1:N
        i = schedule(j,s);
        A = norm(H(:,:,i,j)*V(:,i))^2;
        if A == 0
            continue
        end
        
        B = noise;
        for n = 1:L
            for t = 1:N
                m = schedule(n,t);
                B = B + norm(H(:,:,m,j)*V(:,m))^2;
            end
        end
        
        currentSINR(i) = A/(B-A);
    end
end

end