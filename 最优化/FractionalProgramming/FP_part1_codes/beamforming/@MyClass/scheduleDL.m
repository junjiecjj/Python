function [ schedule ] = scheduleDL( obj, weight, V )

L = obj.numBS;
M = obj.numTxAnte;
N = obj.numRxAnte;
H = obj.chn;
T = obj.numTone;
noise = obj.noise;
association = obj.association;
schedule = zeros(T,L,N);

for j = 1:L
    users_in_cell = find(association==j);
    for z = 1:T
        utility = nan(length(users_in_cell),M);
        
        for index = 1:length(users_in_cell)
            i = users_in_cell(index);
            for s = 1:M
                B = eye(N)*noise;
                for n = 1:L
                    for t = 1:M
                        if n==j && t==s
                            continue
                        end
                        B = B + H(:,:,z,i,n)*V(:,z,n,t)*V(:,z,n,t)'*H(:,:,z,i,n)';
                    end
                end
                sinr = abs(V(:,z,j,s)'*H(:,:,z,i,j)'*(B\H(:,:,z,i,j)*V(:,z,j,s)));
                utility(index,s) = weight(i)*log2(1+sinr);         
            end
        end
        
        matching = Hungarian(utility);
        for s = 1:M
            index = matching(s);
            schedule(z,j,s) = users_in_cell(index);
        end
    end
end

end