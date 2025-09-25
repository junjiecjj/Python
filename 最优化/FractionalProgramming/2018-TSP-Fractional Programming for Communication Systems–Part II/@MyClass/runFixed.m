function [ schedule, U, V ] = runFixed( obj, weight, numIter )

L = obj.numBS;
K = obj.numUser;
M = obj.numTxAnte;
noise = obj.noise;
H = obj.chn;
maxPower = obj.maxPower;
association = obj.association;
schedule = initScheduler(obj, weight);

% initialize
V = nan(M,K);
for i = 1:K
    V(:,i) = ones(M,1)*sqrt(maxPower(i)/M);
end

for iter = 1:numIter
    % beamforming
    [schedule, U, V ] = runBeamforming(obj, weight, schedule, V, 1);
    
    %%
    wrate = nan(K,1);
    for j = 1:L
        usersInCell = find(association==j)';
        
        B = noise;
        for n = 1:L
            if n == j
                continue
            end
            m = schedule(n);
            B = B + norm(H(:,:,m,j)*V(:,m))^2;
        end
        
        for i = usersInCell
            A = norm(H(:,:,i,j)*V(:,i))^2;
            wrate(i) = weight(i)*log2(1+A/B);
        end
    end
    
    for j = 1:L
        usersInCell = find(association==j)';
        [~,index] = max(wrate(usersInCell));
        schedule(j,1) = usersInCell(index);
    end
end
            
end