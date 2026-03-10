function [DOA_deg,DOA_rad] = Gr_SBL_1Bit(x0_one,K,M,L,iter,a,b,G)



%% Paper: A Generalized Sparse Bayesing Learning Algorithm for 1-bit DOA Estimation (2018)

theta = linspace(-pi/2, pi/2, G);
idxR = (0:(M-1))';
d_lambda = 0.5;

A = zeros(M,G);
for ii = 1:G
    A(:,ii) = exp(-1i*2*pi*d_lambda*sin(theta(ii))*idxR);
end

R_Y = real(x0_one);
I_Y = imag(x0_one);
XX = zeros(G,L);


%% Initialization
V_A_0 = ones(M,L); 
Z_A_0 = ones(M,L) + 1j*ones(M,L);
Alpha_0 = ones(G,L);

R_Z_A_0 = real(Z_A_0);
I_Z_A_0 = imag(Z_A_0);

R_V_1 = zeros(M,L);
I_V_1 = zeros(M,L);

R_Z_A_1 = zeros(M,L);
I_Z_A_1 = zeros(M,L);

Z_B_post = zeros(M,L);
V_B_post = zeros(M,L);

for iii = 1:iter


    for jjj = 1:L

        for mmm = 1:M

           % Step 3

            n_real = normpdf(0,R_Z_A_0(mmm,jjj),0.5*(V_A_0(mmm,jjj)+1));
            phi_real = normcdf((R_Y(mmm,jjj)*R_Z_A_0(mmm,jjj))/(sqrt(0.5*(V_A_0(mmm,jjj)+1))));
            R_Z_A_1(mmm,jjj) = R_Z_A_0(mmm,jjj) + ( R_Y(mmm,jjj)*(0.5*V_A_0(mmm,jjj))*(n_real/phi_real) );
            e_z2_real = ( 0.5*V_A_0(mmm,jjj) ) + ( R_Z_A_0(mmm,jjj)^2  ) +  ( R_Y(mmm,jjj)*(0.5*V_A_0(mmm,jjj))* n_real *  ( R_Z_A_0(mmm,jjj) + ( R_Z_A_0(mmm,jjj)/ (1 + V_A_0(mmm,jjj)) )  )   )  / ( phi_real ) ;
            R_V_1(mmm,jjj) =  e_z2_real -  (R_Z_A_1(mmm,jjj)^2);
            if R_V_1(mmm,jjj) < 0

                R_V_1(mmm,jjj) = 10^(-10);
            end


            n_real = normpdf(0,I_Z_A_0(mmm,jjj),0.5*(V_A_0(mmm,jjj)+1));
            phi_real = normcdf((I_Y(mmm,jjj)*I_Z_A_0(mmm,jjj))/(sqrt(0.5*(V_A_0(mmm,jjj)+1))));
            I_Z_A_1(mmm,jjj) = I_Z_A_0(mmm,jjj) + ( I_Y(mmm,jjj)*(0.5*V_A_0(mmm,jjj))*(n_real/phi_real) );
            e_z2_real = ( 0.5*V_A_0(mmm,jjj) ) + ( I_Z_A_0(mmm,jjj)^2  ) +  ( I_Y(mmm,jjj)*(0.5*V_A_0(mmm,jjj))* n_real *  ( I_Z_A_0(mmm,jjj) + ( I_Z_A_0(mmm,jjj)/ (1 + V_A_0(mmm,jjj)) )  )   )  / ( phi_real ) ;
            I_V_1(mmm,jjj) =  e_z2_real -  (I_Z_A_1(mmm,jjj)^2);
            if I_V_1(mmm,jjj) < 0

                I_V_1(mmm,jjj) = 10^(-10);
            end

            
            Z_B_post(mmm,jjj) = R_Z_A_1(mmm,jjj) + 1j*I_Z_A_1(mmm,jjj); 
            V_B_post(mmm,jjj) = R_V_1(mmm,jjj) + I_V_1(mmm,jjj);

        end


        % Step 4:

        v_b_ext = real( 1 ./ ( (1 ./ V_B_post(:,jjj)) - ( 1 ./ V_A_0(:,jjj) )   ) );

        for mmm=1:M
            if v_b_ext(mmm) < 0
                v_b_ext(mmm) = 10^(-10);
            end
        end

        Sig_til = v_b_ext;
        z_b_ext = Sig_til .* ( (Z_B_post(:,jjj) ./ V_B_post(:,jjj) ) - ( Z_A_0(:,jjj) ./ V_A_0(:,jjj) )  );
        y_til = z_b_ext; 


        % Step 5:

        alph_hat = mean(Alpha_0,2)  ;

        Sigma_post_x = ( ( (A')*diag(1./Sig_til)*A ) + diag(alph_hat) )\eye(G);
        mu_post_x = Sigma_post_x*(A')*diag(1./Sig_til)*y_til;


        XX(:,jjj) = mu_post_x;

        % Step 6:
        
        for ggg = 1:G
        Alpha_0 (ggg,jjj) = a / ( b + (abs( mu_post_x(ggg) )^2) + Sigma_post_x(ggg,ggg) ) ;
        end


        % Step 7:

        z_A_post = A*mu_post_x;
        v_A_post = diag( A * Sigma_post_x * (A') );


        % Step 8:

        V_A_0(:,jjj) = real( 1 ./ ( ( 1./v_A_post ) - ( 1./Sig_til  ) ) );
        
        for mmm = 1:M
        if V_A_0(mmm,jjj) < 0
            V_A_0(mmm,jjj) = 10^(-10);
        end


        end
        Z_A_0(:,jjj) = V_A_0(:,jjj) .* ( ( z_A_post./v_A_post ) - (  y_til./Sig_til  )  ) ;


    end

    R_Z_A_0 = real(Z_A_0);
    I_Z_A_0 = imag(Z_A_0);

end

s_til = zeros(G,1);

for iii=1:G
    s_til(iii,1) = norm(XX(iii,:));
end

[~,LOCS]= findpeaks(s_til,'SortStr','descend','NPeaks',K);

if length(LOCS) < K
    LOCS=[LOCS LOCS LOCS];
    LOCS=LOCS(1:K);
end

DOA_rad = sort(theta(LOCS).');
DOA_deg = DOA_rad * (180/pi);



end