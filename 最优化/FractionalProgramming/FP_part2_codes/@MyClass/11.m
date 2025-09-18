function [ schedule, U, V ] = runProposed( obj, weight, numIter )

L = obj.numBS;
K = obj.numUser;
M = obj.numTxAnte;
N = obj.numRxAnte;
noise = obj.noise;
H = obj.chn;
maxPower = obj.maxPower;
association = obj.association;
schedule = initScheduler(obj, weight);

% initialize U & V
V = nan(M,K);
for i = 1:K
    V(:,i) = ones(M,1)*sqrt(maxPower(i)/M);
end

U = nan(N,L,N);
for j = 1:L
    B = noise*eye(N);
    for n = 1:L
        for t = 1:N
            m = schedule(n,t);
            B = B + H(:,:,m,j)*V(:,m)*V(:,m)'*H(:,:,m,j)';
        end
    end

    for s = 1:N
        i = schedule(j,s);
        A = H(:,:,i,j)*V(:,i);
        U(:,j,s) = B\A;
    end
end 

gamma = updateGamma( L, N, schedule, sinr );

% algorithm iterations
for iter = 1:numIter
    sinr = computeCurrentSINR(obj, schedule, U, V);
    y = updateY( L, N, H, schedule, U, V, weight, gamma );
    gamma = updateGamma( L, N, schedule, sinr );
    
	U = updateU( L, N, noise, H, schedule, V, weight, gamma, y );
    [ schedule, V ] = updateScheduleAndV( L, K, M, N, H, maxPower, schedule,...
        U, V, weight, y, gamma, association );
end

end

%%
function [ gamma ] = updateGamma( L, N, schedule, sinr )

gamma = nan(L,N);
for j = 1:L
    for s = 1:N
        i = schedule(j,s);
        gamma(j,s) = sinr(i);
    end
end

end
%%
function [ y ] = updateY( L, N, H, schedule, U, V, weight, gamma )

y = nan(L,N);
for j = 1:L
    for s = 1:N
        i = schedule(j,s);
        y(j,s) = sqrt(weight(i)*(1+gamma(j,s)))/(1+1/gamma(j,s))...
            /norm(U(:,j,s)'*H(:,:,i,j)*V(:,i));
        if isnan(y(j,s))
            y(j,s) = 0;
        end
    end
end

end
%%
function [ U ] = updateU( L, N, noise, H, schedule, V, weight, gamma, y )

U = zeros(N,L,N);

for j = 1:L
    B = noise*eye(N);
    for n = 1:L
        for t = 1:N
            m = schedule(n,t);
            B = B + H(:,:,m,j)*V(:,m)*V(:,m)'*H(:,:,m,j)';
        end
    end
    
    for s = 1:N
        if y(j,s) == 0
            continue
        end
        i = schedule(j,s);
        A = sqrt(weight(i)*(1+gamma(j,s)))*H(:,:,i,j)*V(:,i)/y(j,s);
        U(:,j,s) = B\A;
    end
end
            
end
%%
function [ schedule, V ] = updateScheduleAndV( L, K, M, N, H, maxPower,...
    schedule, U, V, weight, y, gamma, association )

Z = nan(M,K,N);

for i = 1:K
    j = association(i);
    B = zeros(M,M);
    for n = 1:L
        for t = 1:N
            B = B + y(n,t)^2*H(:,:,i,n)'*U(:,n,t)*U(:,n,t)'*H(:,:,i,n);
        end
    end
    
    for s = 1:N
        A = y(j,s)*sqrt(weight(i)*(1+gamma(j,s)))*H(:,:,i,j)'*U(:,j,s);
        tempZ = B\A;
        
        if norm(tempZ)^2 <= maxPower(i)
            Z(:,i,s) = tempZ;
            continue
        end
        
        % bisection search on opt mu
        muLeft = 0;
        muRight = 1;
        while 1
            tempZ = (B+muRight*eye(M))\A;
            if norm(tempZ)^2 <= maxPower(i)
                break
            end
            muRight = muRight*10;
        end
        
        while 1
            mu = (muLeft+muRight)/2;
            tempZ = (B+mu*eye(M))\A;
            if abs(norm(tempZ)^2-maxPower(i)) < maxPower(i)/1e3
                Z(:,i,s) = tempZ;
                break
            end
            
            if norm(tempZ)^2 > maxPower(i)
                muLeft = mu;
            else
                muRight = mu;
            end
        end
    end
end

% assignment problem
for j = 1:L
    usersInCell = find(association==j)';
    numUsersInCell = length(usersInCell); % number of users
    
    utility = nan(numUsersInCell,N);
    for q = 1:numUsersInCell
        i = usersInCell(q);
        for s = 1:N
            utility(q,s) = weight(i)*log(1+gamma(j,s)) - weight(i)*gamma(j,s)...
                + 2*y(j,s)*sqrt(weight(i)*(1+gamma(j,s)))*norm(U(:,j,s)'*H(:,:,i,j)*Z(:,i,s));
            for n = 1:L
                for t = 1:N
                    utility(q,s) = utility(q,s) - y(n,t)^2*norm(U(:,n,t)'*H(:,:,i,n)*Z(:,i,s))^2;
                end
            end
        end
    end
    
    assignment = Hungarian(utility);
    for s = 1:N
        i = usersInCell(assignment(s));
        schedule(j,s) = i;
        V(:,i) = Z(:,i,s);
    end
end
            
end