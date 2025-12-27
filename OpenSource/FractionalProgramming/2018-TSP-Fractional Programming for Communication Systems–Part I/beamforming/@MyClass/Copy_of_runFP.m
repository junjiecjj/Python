function [ instRate ] = runFP( obj, weight, numIter, V )

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
    
%     VV = V;
%     for j = 1:L
%         for z = 1:T
%             i = schedule(z,j,1);
%             A = abs(y(1,:,z,j)*H(:,:,z,i,j)/(1+sinr(z,j)));
%             B = 0;
%             for n = 1:L
%                 if n == j
%                     continue
%                 end
%                 m = schedule(z,n,1);
%                 B = B + abs(y(1,:,z,n)*H(:,:,z,m,j))^2/(1+sinr(z,n));
%             end
%             VV(:,z,j) = min(A/B,maxPower(j));
%         end
%     end
%     
%     V = VV;
    
    %%   
%     theta = nan(L*M,1);
%     for j = 1:L
%         for s = 1:M
%             i = schedule(j,s);
%             theta(M*(j-1)+s) = 1 + 2*real(y(1,:,j,s)*H(:,:,i,j)*V(:,j,s)) - noise*norm(y(1,:,j,s))^2;
%             for n = 1:L
%                 for t = 1:M
%                     if n == j && t == s
%                         continue
%                     end
%                     theta(M*(j-1)+s) = theta(M*(j-1)+s) - abs(y(1,:,j,s)*H(:,:,i,n)*V(:,n,t))^2;
%                 end
%             end           
%         end
%     end
%     sum_rate = sum(obj.bandwidth*log2(theta));
    
    A = zeros(L*M,M,T,L,M);
    for j = 1:L
        for s = 1:M
            for z = 1:T
                i = schedule(z,j,s);
                for n = 1:L
                    for t = 1:M
                        if n==j && t==s
                            continue
                        end
                        A(M*(n-1)+t,:,z,j,s) = y(1,:,z,j,s)*H(:,:,z,i,n);
                    end
                end
            end
        end
    end
                

%     for j = 1:L
%         for s = 1:N
%             i = schedule(j,s);
%             theta(N*(j-1)+s) = 1 + 2*real(y(1,:,j,s)*H(:,:,i,j)*V(:,i)) - sum_square_abs(y(1,:,j,s))*noise;
%             for n = 1:L
%                 for t = 1:N
%                     if n == j && t == s
%                         continue
%                     end
%                     m = schedule(n,t);
%                     theta(N*(j-1)+s) = theta(N*(j-1)+s) - abs(y(1,:,j,s)*H(:,:,i,n)*V(:,m))^2;
%                 end
%             end
%         end
%     end 
    
%     theta = nan(L*M,T);
    F = nan(M,L*M,T); % reshape V and weight
    for j = 1:L
        for s = 1:M
            for z = 1:T
                F(:,M*(j-1)+s,z) = V(:,z,j,s);
            end
        end
    end
%     for j = 1:L
%         for s = 1:M
%             for z = 1:T
%                 i = schedule(z,j,s);
%                 theta(M*(j-1)+s,z) = 1 + 2*real(y(1,:,z,j,s)*H(:,:,z,i,j)*F(:,M*(j-1)+s,z)) - sum_square_abs(y(1,:,z,j,s))*noise...
%                 - sum_square_abs(diag(A(:,:,z,j,s)*F(:,:,z)));
%             end
%         end
%     end    
%     sum_rate = sum(sum(obj.bandwidth*log2(theta)));

%     theta = nan(L,M,T);
%     for j = 1:L
%         for s = 1:M
%             for z = 1:T
%                 theta(j,s,z) = log2(1 + 2*real(y(1,:,z,j,s)*H(:,:,z,schedule(z,j,s),j)*F(:,M*(j-1)+s,z)) - sum_square_abs(y(1,:,z,j,s))*noise ...
%                     - sum_square_abs(diag(A(:,:,z,j,s)*F(:,:,z))));
%             end
%         end
%     end  
%     sum(sum(sum(theta)))

    cvx_begin
        variable F(M,L*M,T) complex 
        variable theta(L*M,T)
        maximize ( sum(sum(log(theta))) )%( sum(sum(theta)) )%( W*log(theta)' ) %geo_mean(theta)%
        subject to
            for j = 1:L
                sum(sum_square_abs(F(:,j,:))) <= maxPower(j) % use this when numTxAnte==1 and numTone==1
%                 sum(sum(sum(square_abs(F(:,M*(j-1)+1:M*(j-1)+M,:))))) <= maxPower(j)
                for s = 1:M
                    for z = 1:T
                        1 + 2*real(y(1,:,z,j,s)*H(:,:,z,schedule(z,j,s),j)*F(:,M*(j-1)+s,z)) - sum_square_abs(y(1,:,z,j,s))*noise ...
                            - sum_square_abs(diag(A(:,:,z,j,s)*F(:,:,z))) >= theta(M*(j-1)+s,z)%exp(theta(M*(j-1)+s,z))
                    end
                end
            end
    cvx_end
    
    V = nan(M,T,L,M);
    for j = 1:L
        for s = 1:M
            for z = 1:T
                V(:,z,j,s) = F(:,M*(j-1)+s,z);
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