function [ instRate ] = runWMMSE( obj, weight, numIter, V )

L = obj.numBS;
M = obj.numTxAnte;
N = obj.numRxAnte;
T = obj.numTone;
noise = obj.noise;
H = obj.chn;
maxPower = obj.maxPower;

global convergeWMMSE
convergeWMMSE = nan(numIter,1);

schedule = scheduleDL( obj, weight, V );

for iter = 1:numIter
    fprintf('WMMSE: %d\n', iter)  
    [instRate, sinr] = computeInstRate( obj, schedule, V );
    convergeWMMSE(iter) = sum(weight.*instRate);
    
    if 1;%mod(iter,5)==1
    W = 1 + sinr;
    end
    % update U
    U = nan(N,T,L,M);
    for j = 1:L
        for s = 1:M
            for z = 1:T
                i = schedule(z,j,s);
                A = H(:,:,z,i,j)*V(:,z,j,s);
                B = eye(N)*noise;
                for n = 1:L
                    for t = 1:M
                        B = B + H(:,:,z,i,n)*V(:,z,n,t)*V(:,z,n,t)'*H(:,:,z,i,n)';
                    end
                end
                U(:,z,j,s) = B\A;
            end
        end
    end
    
    % update W
%     W = 1 + sinr;
    
    % update V
    V = nan(M,T,L,M);
    for j = 1:L
        A = nan(M,T,M);
        B = nan(M,M,T,M);
        for s = 1:M
            for z = 1:T
                i = schedule(z,j,s);
                A(:,z,s) = weight(i)*W(z,j,s)*H(:,:,z,i,j)'*U(:,z,j,s);
                B(:,:,z,s) = zeros(M,M);
                for n = 1:L
                    for t = 1:M
                        m = schedule(z,n,t);
                        B(:,:,z,s) = B(:,:,z,s) + ...
                            weight(m)*W(z,n,t)*H(:,:,z,m,j)'*U(:,z,n,t)*U(:,z,n,t)'*H(:,:,z,m,j);
                    end
                end
                V(:,z,j,s) = B(:,:,z,s)\A(:,z,s);
            end
        end
        
        if sum(sum(sum(square_abs(V(:,:,j,:))))) <= maxPower(j)
            continue
        end
        
        % bisection search on opt mu
        muLeft = 0;
        muRight = 1;
        while 1
            for s = 1:M
                for z = 1:T
                    V(:,z,j,s) = (B(:,:,z,s)+eye(M)*muRight)\A(:,z,s);
                end
            end
            if sum(sum(sum(square_abs(V(:,:,j,:))))) <= maxPower(j)
                break
            end
            muRight = muRight*10;
        end
        
        while 1
            mu = (muLeft+muRight)/2;
            for s = 1:M
                for z = 1:T
                    V(:,z,j,s) = (B(:,:,z,s)+eye(M)*mu)\A(:,z,s);
                end
            end
            
            abs(sum(sum(sum(square_abs(V(:,:,j,:))))) - maxPower(j))
            if abs( sum(sum(sum(square_abs(V(:,:,j,:))))) - maxPower(j) ) < maxPower(j)/1e8
                break
            end
                
            if sum(sum(sum(square_abs(V(:,:,j,:))))) > maxPower(j)
                muLeft = mu;
            else
                muRight = mu;
            end
        end
    end
end

end