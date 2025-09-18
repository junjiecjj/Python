function [ schedule, U, V ] = runWMMSE( obj, weight, numIter )

L = obj.numBS;
K = obj.numUser;
M = obj.numTxAnte;
N = obj.numRxAnte;
noise = obj.noise;
H = obj.chn;
maxPower = obj.maxPower;
association = obj.association;
schedule = nan(L,N);

% initialize
V = nan(M,K);
for i = 1:K
    V(:,i) = ones(M,1)*sqrt(maxPower(i)/M);
end
UU = zeros(N,K); % U is BS-specific, UU is user-specific
W = zeros(1,K);

for iter = 1:numIter
    % update U
    for j = 1:L
        usersInCell = find(association==j)';
        B = noise*eye(N);
        for m = 1:K
            B = B + H(:,:,m,j)*V(:,m)*V(:,m)'*H(:,:,m,j)';
        end
        %%
        for i = usersInCell
            A = H(:,:,i,j)*V(:,i);
            UU(:,i) = B\A;
        end
    end
    
    % udpate W
    for i = 1:K
        j = association(i);
        MSE = real(1 - UU(:,i)'*H(:,:,i,j)*V(:,i));
        W(i) = 1/MSE;
    end
    
    % update V
    for i = 1:K
        if norm(V(:,i))^2 < maxPower(i)/1e3
            V(:,i) = zeros(M,1);
            continue
        end
        
        j = association(i);
        A = weight(i)*W(i)*H(:,:,i,j)'*UU(:,i);
        B = zeros(M,M);
        for m = 1:K
            n = association(m);
            B = B + weight(m)*W(m)*H(:,:,i,n)'*UU(:,m)*UU(:,m)'*H(:,:,i,n);
        end
        tempV = B\A;
        V(:,i) = min(tempV, sqrt(maxPower(i)));
    end
end

%%
power = nan(K,1);
for i = 1:K
    power(i) = norm(V(:,i))^2;
end

U = nan(N,L,N);
for j = 1:L
    usersInCell = find(association==j)';
    [~,index] = sort(power(usersInCell),'descend');
    sortedUsers = usersInCell(index);
    for s = 1:N
        i = sortedUsers(s);
        schedule(j,s) = i;
        U(:,j,s) = UU(:,i);
    end
end
            
end