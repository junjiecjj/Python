function [ schedule, U, V ] = runBeamforming( obj, weight, schedule, V, numIter )

L = obj.numBS;
M = obj.numTxAnte;
N = obj.numRxAnte;
noise = obj.noise;
H = obj.chn;
maxPower = obj.maxPower;

U = nan(N,L,N);
W = nan(1,L,N);

for iter = 1:numIter
    % update U
    for j = 1:L
        for s = 1:N
            i = schedule(j,s);
            A = H(:,:,i,j)*V(:,i);
            
            B = noise*eye(N);
            for n = 1:L
                for t = 1:N
                    m = schedule(n,t);
                    B = B + H(:,:,m,j)*V(:,m)*V(:,m)'*H(:,:,m,j)';
                end
            end
            
            U(:,j,s) = B\A;
        end
    end
    
    % udpate W
    for j = 1:L
        for s = 1:N
            i = schedule(j,s);
            MSE = real(1 - U(:,j,s)'*H(:,:,i,j)*V(:,i));
            W(j,s) = 1/MSE;
        end
    end
    
    % update V
    for j = 1:L
        s = 1;
        i = schedule(j,s);
        A = weight(i)*W(j,s)*H(:,:,i,j)'*U(:,j,s);
        B = zeros(M,M);
        for n = 1:L
            for t = 1:N
                m = schedule(n,t);
                B = B + weight(m)*W(n,t)*H(:,:,i,n)'*U(:,n,t)*U(:,n,t)'*H(:,:,i,n);
            end
        end
        tempV = B\A;

        V(:,i) = min(tempV, sqrt(maxPower(i)));
    end
end
      
end