function V_RF = change_V_RF()
    global N N_RF
    global H_t V_RF
    A = cell(1,9);
    B = cell(1,9);
    D = cell(1,9);
    zeta_B = zeros(N,N_RF);
    zeta_D = zeros(N,N_RF);
    eta_B = zeros(N,N_RF);
    eta_D = zeros(N,N_RF);
    c = zeros(N,N_RF);
    z = zeros(N,N_RF);
    phi = zeros(N,N_RF);
    for j = 1:N_RF
        V_RF_bar = V_RF;
        V_RF_bar(:,j) = [];
        A{j} = H_t * V_RF_bar*(V_RF_bar')* H_t';
        % 计算B_j,D_j,用于计算zeta和eta
        B{j} = H_t'* (A{j}^(-2)) * H_t;
        D{j} = H_t'* (A{j}^(-1)) * H_t;
        for i = 1:N
            % 计算zeta和eta
            temp = 0;
            for m = 1:1:N
                if( m~=i )
                    for n = 1:1:N
                        if(n~=i)
                            temp = temp + conj(V_RF(m,j))*B{j}(m,n)*V_RF(n,j);
                        end
                    end
                end
            end
            zeta_B(i,j)= B{j}(i,i)+2*real(temp);
            temp = 0;
            for m = 1:1:N
                if( m~=i )
                    for n = 1:1:N
                        if(n~=i)
                            temp = temp + conj(V_RF(m,j))*D{j}(m,n)*V_RF(n,j);
                        end
                    end
                end
            end
            zeta_D(i,j)= D{j}(i,i)+2*real(temp);
            V_B = B{j} * V_RF;
            V_D = D{j} * V_RF;
            eta_B(i,j) = V_B(i,j) - B{j}(i,i)*V_RF(i,j);
            eta_D(i,j) = V_D(i,j) - D{j}(i,i)*V_RF(i,j);
            % 计算theta_1，theta_2
            c(i,j) = (1 + zeta_D(i,j)) * eta_B(i,j) - zeta_B(i,j) * eta_D(i,j);
            z(i,j) = imag( 2 * eta_B(i,j) * eta_D(i,j));
            phi(i,j) = 0;
            tt = asin(imag(c(i,j))/abs(c(i,j)));
            if(real(c(i,j))>=0) 
                phi(i,j) = tt; 
            else 
                phi(i,j) = pi - tt; 
            end
            theta_1 = -phi(i,j) + asin(z(i,j)/abs(c(i,j)));
            theta_2 = pi - phi(i,j) - asin(z(i,j)/abs(c(i,j)));
            % 判断最优theta
            V_RF1 = exp(-1j * theta_1);
            V_RF2 = exp(-1j * theta_2);
            f1 = N * trace(A{j}^(-1)) -  N * ( zeta_B(i,j) + 2 * real(conj(V_RF1)*eta_B(i,j)))/ (1+zeta_D(i,j)+2*real(conj(V_RF1)*eta_D(i,j)));
            f2 = N * trace(A{j}^(-1)) - N * ( zeta_B(i,j) + 2 * real(conj(V_RF2)*eta_B(i,j)))/ (1+zeta_D(i,j)+2*real(conj(V_RF2)*eta_D(i,j)));
            if(f1 < f2) 
                theta_opt = theta_1; 
            else 
                theta_opt = theta_2;
            end
            V_RF(i,j) = exp(-1j*theta_opt);
        end
    end
end







