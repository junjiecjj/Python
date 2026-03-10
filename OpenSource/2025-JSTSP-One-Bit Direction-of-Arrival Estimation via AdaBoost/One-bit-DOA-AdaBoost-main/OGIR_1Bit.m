function [DOA_deg,DOA_rad] = OGIR_1Bit(x0_one,K,M,N,T,t0,eps,r0,p0,K_tilde)



%% Paper: An Off-Grid Iterative Reweighted Approach to One-bit Direction of Arrival Estimation (2023)

theta_0 = linspace(-pi/2, pi/2, K);
idxR = (0:(M-1))';
d_lambda = 0.5;
Alpha=[];
A = zeros(M,K);
for ii = 1:K
    A(:,ii) = exp(-1i*2*pi*d_lambda*sin(theta_0(ii))*idxR);
end

Res = ((round(180/K)*(pi/180))/2) * ones(K,1);

R_Y = real(x0_one);
I_Y = imag(x0_one);

%% Initialization

X_0 = ((A')*x0_one)/(norm((A')*x0_one,'fro'));
gamma = r0*norm(X_0,'fro');
h_1_prime = zeros(M,N);
count = 0;
ter = 0;

while ter == 0

    % Step 2:
    D = (R_Y .* real( A*X_0 ) ) + 1j*(I_Y .* imag( A*X_0 ) );
    for mmm=1:M
        for nnn=1:N
        [hh_1_R] = hh_1_prime(real(D(mmm,nnn)));
        [hh_1_I] = hh_1_prime(imag(D(mmm,nnn)));
        h_1_prime(mmm,nnn) = hh_1_R +1j*hh_1_I;


        end
    end

    V_tilde = D - h_1_prime;
    V_t = (R_Y .* real( V_tilde ) ) + 1j*(I_Y .* imag( V_tilde ) );
    pp = zeros(K,1); 
    for kkk=1:K
       pp(kkk) = (norm(X_0(kkk,:))^2) + eps;
    end
    P = diag(pp)/(2*gamma);

    B = ( eye(M) + A*P*(A') )\eye(M);
    X_1 = ( P - P*(A')*B*A*P )*( (A')*V_t  );
    X_1 = (X_1)/(norm(X_1,'fro'));

    % Steps 3 and 4:
    if count >= t0

        pp = zeros(K,1); 
        for kkk=1:K
           pp(kkk) = (norm(X_1(kkk,:))^2);
        end
        p_m = max(pp);

        p_k = zeros(K,1);
        for kkk=1:K
           p_k(kkk) = (norm(X_1(kkk,:))^2)/p_m;
        end

        IND = find(p_k >= p0);

        theta_bar = theta_0;

        for kkk = 1:length(IND)
            si = (M*(M-1)*0.5)*conj(X_1(IND(kkk),:))*(X_1(IND(kkk),:).');

            A_bar = zeros(M,K);
            for ii = 1:K
              A_bar(:,ii) = exp(-1i*2*pi*d_lambda*sin(theta_bar(ii))*idxR);
            end
            A_minus = A_bar;
            A_minus(:,IND(kkk)) = []; 
            X_minus = X_1;
            X_minus(IND(kkk),:) = [];

            Mm = 0:M-1;
            SI = ( X_1(IND(kkk),:)*(X_minus')*(A_minus') + X_1(IND(kkk),:)*(V_t') ).*Mm;

            Coef = [ SI(M:-1:2) si];
            RoT = roots(Coef);
            alpha = asin(-(1/pi)*angle(RoT));

            if IND(kkk) == 1
                JJ =[theta_bar(IND(kkk))    theta_bar(IND(kkk)) + Res(IND(kkk))  ];
                Ind_alph = find( (theta_bar(IND(kkk)) < alpha ) &  (alpha < theta_bar(IND(kkk)) + Res(IND(kkk)) ) );
                alpha = alpha(Ind_alph);
            elseif IND(kkk) == K
                JJ =[theta_bar(IND(kkk))  theta_bar(IND(kkk)) - Res(IND(kkk)) ];
                Ind_alph = find( ( theta_bar(IND(kkk)) - Res(IND(kkk)) < alpha ) &  (alpha < theta_bar(IND(kkk)) ) );
                alpha = alpha(Ind_alph);
            else
                JJ =[theta_bar(IND(kkk))  (theta_bar(IND(kkk)) - Res(IND(kkk))) (theta_bar(IND(kkk)) + Res(IND(kkk))) ];
                Ind_alph = find( ( (theta_bar(IND(kkk)) - Res(IND(kkk)))  < alpha ) &  (alpha < (theta_bar(IND(kkk)) + Res(IND(kkk))) ) );
                alpha = alpha(Ind_alph);
            end
            
            Thet = union(alpha,JJ);
            LL = zeros(length(Thet),1);

            for iii=1:length(Thet)
               aa = exp(-1i*2*pi*d_lambda*sin(Thet(iii))*idxR);
               LL(iii,1) = (conj(X_1(IND(kkk),:))*(X_1(IND(kkk),:).')*(aa')*aa) + X_1(IND(kkk),:)*(X_minus')*(A_minus')*aa + 2*real( X_1(IND(kkk),:)*(V_t')*aa );
            end

            [~,in_ma]=min(LL);
            tt = Thet(in_ma);

           
        
            theta_bar(IND(kkk)) = tt;
           

           Res(IND(kkk)) = Res(IND(kkk))/2;
        end

      
        theta_1 = theta_bar;
        
    else
        theta_1 = theta_0;
    end

     

    % Step 5:
    gamma = r0*norm(X_1,'fro');

    if count >= T
        ter = 1;
    end

    der = (norm(X_1 - X_0,'fro')^2)/(norm(X_0,'fro')^2);

    
    per = (norm(theta_1 - theta_0)^2)/(norm(theta_0)^2);
    Alpha = [Alpha; der];
    

    
         if der < (10^(-6))
             if per < (10^(-6))    
                ter = 1;
             end
         end

   X_0 = X_1;
   theta_0 = theta_1;

   A = zeros(M,K);
   for ii = 1:K
      A(:,ii) = exp(-1i*2*pi*d_lambda*sin(theta_0(ii))*idxR);
   end

count = count + 1;
end

s_til = zeros(K,1);

for iii=1:K
    s_til(iii,1) = norm(X_1(iii,:));
end

[~,LOCS]= findpeaks(s_til,'SortStr','descend','NPeaks',K_tilde);

if length(LOCS) < K_tilde
    LOCS=[LOCS LOCS LOCS];
    LOCS=LOCS(1:K_tilde);
end

DOA_rad = sort(theta_1(LOCS).');
DOA_deg = DOA_rad * (180/pi);

end



function [hh_1] = hh_1_prime(x0)

hh_1 = -exp( - (x0^2)/2 )/(sqrt(2*pi)*normcdf(x0));


end