function [ f,time ] = CQT( iter_num, P, N, d, L, A, B, X )
Y = nan(d,L,N);

f = nan(iter_num+1,1);
time = nan(iter_num,1);

for iter = 1:iter_num
    f(iter) = orig_fun( N, d, A, B, X );
    tic
    for n = 1:N
        D = eye(d);

        for m = 1:N
            BX = B(:,:,m,n)*X(:,:,m);
            D = D + BX*BX';
        end

        numerator = A(:,:,n)*X(:,:,n);
        Y(:,:,n) = D\numerator;
    end

    for n = 1:N
        D = zeros(d,d);

        for m = 1:N
            BY = B(:,:,n,m)'*Y(:,:,m);
            D = D + BY*BY';
        end

        numerator = A(:,:,n)'*Y(:,:,n);
        X(:,:,n) = D\numerator;
        if trace(X(:,:,n)'*X(:,:,n)) < P
            continue
        end

        mu_right = 1;
        while trace(X(:,:,n)'*X(:,:,n)) > P
            mu_right = 10*mu_right;
            tempD = D + mu_right*eye(d);
            X(:,:,n) = tempD\numerator;
        end

        mu_left = 0;
        while abs(trace(X(:,:,n)'*X(:,:,n))-P)/P > 1e-10
            mu = (mu_left + mu_right)/2;
            tempD = D + mu*eye(d);
            X(:,:,n) = tempD\numerator;

            if trace(X(:,:,n)'*X(:,:,n)) > P
                mu_left = mu;
            else
                mu_right = mu;
            end
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