function [ f,time ] = EQT_Polyak( iter_num, P, N, d, L, A, B, X )
Y = nan(d,L,N);

f = nan(iter_num+1,1);
time = nan(iter_num,1);
for iter = 1:iter_num

    f(iter) = orig_fun( N, d, A, B, X );
    tic
    for n = 1:N % update y
        D = zeros(d,d);
        D = eye(d);

        for m = 1:N

            BX = B(:,:,m,n)*X(:,:,m);
            D = D + BX*BX';
        end

        numerator = A(:,:,n)*X(:,:,n);
        Y(:,:,n) = D\numerator;
    end

    T = X;
    prev_X = X;

    %%
    lambda = nan(N,1);
    for n = 1:N
        D = zeros(d,d);
        for m = 1:N
            BY = B(:,:,n,m)'*Y(:,:,m);
            D = D + BY*BY';
        end

        lambda(n) = real(eigs(D,1));

        gradient = A(:,:,n)'*Y(:,:,n) - D*T(:,:,n);
        X(:,:,n) = T(:,:,n) + 1/lambda(n)*gradient;

        X(:,:,n) = X(:,:,n) + 0.5*(X(:,:,n)-prev_X(:,:,n));

        if trace(X(:,:,n)'*X(:,:,n)) > P
            alpha = sqrt(trace(X(:,:,n)'*X(:,:,n))/P);
            X(:,:,n) = X(:,:,n)/alpha;
        end
    end
    t_run = toc;
    if iter>1
        time(iter) = time(iter-1)+t_run;
    else
        time(iter) = t_run;
    end
end
f(iter+1) = orig_fun( N, d, A, B, X );
end