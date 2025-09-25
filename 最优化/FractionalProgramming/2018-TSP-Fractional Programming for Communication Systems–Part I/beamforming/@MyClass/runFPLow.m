function [ instRate ] = runFPLow( obj, weight, numIter, V )

L = obj.numBS;
K = obj.numUser;
M = obj.numTxAnte;
N = obj.numRxAnte;
T = obj.numTone;
noise = obj.noise;
H = obj.chn;
maxPower = obj.maxPower;

global convergeFP
convergeFP = nan(numIter,1);

schedule = scheduleDL( obj, weight, V );

for iter = 1:numIter
    fprintf('FP: %d\n', iter)
    [ instRate, sinr ] = computeInstRate( obj, schedule, V );
    convergeFP(iter) = sum(weight.*instRate);
    
    [ y ] = updateY( L, T, M, N, noise, H, schedule, V );
    
    for j = 1:L
        A = nan(1,M,T,M);
        B = nan(M,M,T,M);
        for z = 1:T
            for s = 1:M
                i = schedule(z,j,s);
                A(1,:,z,s) = y(1,:,z,j,s)*H(:,:,z,i,j);
                B(:,:,z,s) = zeros(M,M);
                for n = 1:L
                    if n==j
                        continue
                    end
                    m = schedule(z,n,s);
                    B(:,:,z,s) = B(:,:,z,s) + abs(y(1,:,z,n,s)*H(:,:,z,m,j))^2;
                end
                V(:,z,j,s) = B(:,:,z,s)\A(1,:,z,s);
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
                    V(:,z,j,s) = (B(:,:,z,s)+eye(M)*muRight)\A(1,:,z,s);
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
                    V(:,z,j,s) = (B(:,:,z,s)+eye(M)*mu)\A(1,:,z,s);
                end
            end
            
            if abs(sum(sum(sum(square_abs(V(:,:,j,:))))) - maxPower(j)) <= maxPower(j)/1e8
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

function [ y ] = updateY( L, T, M, N, noise, H, schedule, V )

y = nan(1,N,T,L,M);
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
            
            y(1,:,z,j,s) = V(:,z,j,s)'*H(:,:,z,i,j)'/B;
        end
    end
end

end