function [ schedule, V ] = runProposed( obj, weight, numIter )

L = obj.numBS;
K = obj.numUser;
M = obj.numTxAnte;
N = obj.numRxAnte;
H = obj.chn;
maxPower = obj.maxPower;
association = obj.association;
schedule = initScheduler(obj, weight);

% initialize U & V
V = nan(M,K);
for i = 1:K
    V(:,i) = ones(M,1)*sqrt(maxPower(i)/M);
end

% algorithm iterations
for iter = 1:numIter
    sinr = computeCurrentSINR(obj, schedule, V);
    gamma = updateGamma( L, N, schedule, sinr );
    y = updateY( L, N, H, schedule, V, weight, gamma );
    [ schedule, V ] = updateScheduleAndV( L, K, H, maxPower, schedule,...
        V, weight, y, gamma, association );
end

end

%%
function [ gamma ] = updateGamma( L, N, schedule, sinr )

gamma = nan(L,N);
for j = 1:L
    i = schedule(j,1);
    gamma(j,1) = sinr(i);
end

end
%%
function [ y ] = updateY( L, N, H, schedule, V, weight, gamma )

y = nan(L,N);
for j = 1:L
    s = 1;
    i = schedule(j,1);
    y(j,s) = sqrt(weight(i)*(1+gamma(j,1))) / (1+1/gamma(j,1))...
        /norm(H(:,:,i,j)*V(:,i));
    if isnan(y(j,1))
        y(j,1) = 0;
    end
end

end

%%
function [ schedule, V ] = updateScheduleAndV( L, K, H, maxPower,...
    schedule, V, weight, y, gamma, association )

for i = 1:K
    j = association(i);
    B = 0;
    for n = 1:L
        B = B + y(n,1)^2*norm(H(:,:,i,n))^2;
    end
    
    A = y(j,1)*sqrt(weight(i)*(1+gamma(j,1)))*norm(H(:,:,i,j));
    
    V(:,i) = min(A/B, sqrt(maxPower(i)));
end

t = nan(K,1);
for i = 1:K
    j = association(i);
    t(i) = weight(i)*log(1+gamma(j,1)) - weight(i)*gamma(j,1)...
        + 2*y(j,1)*sqrt(weight(i)*(1+gamma(j,1)))*norm(H(:,:,i,j)*V(:,i));
    for n = 1:L
        t(i) = t(i) - y(n,1)^2*norm(H(:,:,i,n)*V(:,i))^2;
    end
end

for j = 1:L
    usersInCell = find(association==j)';
    [~, index] = max(t(usersInCell));
    schedule(j,1) = usersInCell(index);
end
            
end