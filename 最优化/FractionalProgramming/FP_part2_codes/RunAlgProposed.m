function [ schedule, V, U ] = RunAlgProposed( L, K, M, N, NOISE, H, Xmax, schedule, V, w, nIter )

global association

U = NaN(N,L,N);
for j = 1:L
    B = NOISE*eye(N);
    for n = 1:L
        for t = 1:N
            m = schedule(n,t);
            B = B + H(:,:,m,j)*V(:,m,t)*V(:,m,t)'*H(:,:,m,j)';
        end
    end
    
    for s = 1:N
        i = schedule(j,s);
        A = H(:,:,i,j)*V(:,i,s);
        U(:,j,s) = B\A;
    end
end 

for iter = 1:nIter
    gamma = UpdateGamma( L, K, M, N, NOISE, H, schedule, U, V );
    y = UpdateY( L, K, M, N, NOISE, H, schedule, U, V, w, gamma );
	U = UpdateU( L, K, M, N, NOISE, H, Xmax, schedule, V, w, gamma, y );
    [ schedule, V ] = UpdateScheduleAndV( L, K, M, N, NOISE, H, Xmax, schedule, U, w, y, gamma, association );
end

end

%%
function [ gamma ] = UpdateGamma( L, K, M, N, NOISE, H, schedule, U, V )

gamma = NaN(L,N);
SINR = ComputeSINR(L, K, N, NOISE, H, schedule, V, U);
for j = 1:L
    for s = 1:N
        i = schedule(j,s);
        gamma(j,s) = SINR(i);
    end
end

end

%%
function [ y ] = UpdateY( L, K, M, N, NOISE, H, schedule, U, V, w, gamma )

y = nan(L,N);

for j = 1:L
    for s = 1:N
        i = schedule(j,s);
        y(j,s) = 1/(1+1/gamma(j,s))*sqrt(w(i)*(1+gamma(j,s)))/norm(U(:,j,s)'*H(:,:,i,j)*V(:,i,s));
        if isnan(y(j,s))
            y(j,s) = 0;
        end
    end
end

end

%%
function [ U ] = UpdateU( L, K, M, N, NOISE, H, Xmax, schedule, V, w, gamma, y )

U = zeros(N,L,N);

for j = 1:L
    B = NOISE*eye(N);
    for n = 1:L
        for t = 1:N
            m = schedule(n,t);
            B = B + H(:,:,m,j)*V(:,m,t)*V(:,m,t)'*H(:,:,m,j)';
        end
    end
    
    for s = 1:N
        if y(j,s) == 0
            continue
        end
        i = schedule(j,s);
        A = sqrt(w(i)*(1+gamma(j,s)))*H(:,:,i,j)*V(:,i,s)/y(j,s);
        U(:,j,s) = B\A;
    end
end    

if sum(sum(sum(isnan(U)))) > 0
    kk
end
            
end

%%
function [ schedule, V ] = UpdateScheduleAndV( L, K, M, N, NOISE, H, Xmax, schedule, U, w, y, gamma, association )

V = NaN(M,K,N);

for i = 1:K
    j = association(i);
    B = zeros(M,M);
    for n = 1:L
        for t = 1:N
            B = B + y(n,t)^2*H(:,:,i,n)'*U(:,n,t)*U(:,n,t)'*H(:,:,i,n);
        end
    end
    
    for s = 1:N
        A = y(j,s)*sqrt(w(i)*(1+gamma(j,s)))*H(:,:,i,j)'*U(:,j,s);
        V_try = B\A;
        %%
        if norm(V_try)^2 <= Xmax(i)
            V(:,i,s) = V_try;
            continue
        end
        %%
        mu_left = 0;
        mu_right = 1;
        while 1
            V_try = (B+mu_right*eye(M))\A;
            if norm(V_try)^2 <= Xmax(i)
                break
            end
            mu_right = mu_right*10;
        end
        %%
        while 1
            mu = (mu_left+mu_right)/2;
            V_try = (B+mu*eye(M))\A;
            if abs(norm(V_try)^2-Xmax(i)) < Xmax(i)/1e3
                V(:,i,s) = V_try;
                break
            end
            %%%
            if norm(V_try)^2 > Xmax(i)
                mu_left = mu;
            else
                mu_right = mu;
            end
        end
    end
end

% return

%% solving assignment problem in each BS
%% !!! we assume NU > M !!!
for j = 1:L
    users_in_cell = find(association==j)';
    NU = length(users_in_cell); % number of users
    %
    A = nan(NU,N);
    for a = 1:NU
        i = users_in_cell(a);
        for s = 1:N
            A(a,s) = w(i)*log(1+gamma(j,s)) - w(i)*gamma(j,s) + 2*y(j,s)*sqrt(w(i)*(1+gamma(j,s)))...
                *norm(U(:,j,s)'*H(:,:,i,j)*V(:,i,s));
            for n = 1:L
                for t = 1:N
                    A(a,s) = A(a,s) - y(n,t)^2*norm(U(:,n,t)'*H(:,:,i,n)*V(:,i,s))^2;
                end
            end
        end
    end
    %
    [ assignment ] = Hungarian(A);
    for s = 1:N
        schedule(j,s) = users_in_cell(assignment(s));
    end
end
            
end