








% Eq.(24)-(29)
function Vrf = updateVRF(N, Nrf, Htilde, Vrf)
    epsilon = 0.001;
    diff = 1;
    fVrf_old = N * trace(pinv(Htilde * (Vrf * Vrf') * Htilde'));
    it = 0;
    while  (diff > epsilon) & (it < 30)
        it = it + 1;
        for j = 1:Nrf
            Vrfj = Vrf(:, [1:j-1, j+1:end]);
            Aj = Htilde *(Vrfj * Vrfj') * Htilde';
            Bj = Htilde' * Aj^(-2) * Htilde;
            Dj = Htilde' * Aj^(-1) * Htilde;
            for i = 1:N
                % 计算zeta和eta
                temp = 0;
                for m = 1:1:N
                    if( m~=i )
                        for n = 1:1:N
                            if(n~=i)
                                temp = temp + conj(Vrf(m,j))*Bj(m,n)*Vrf(n,j);
                            end
                        end
                    end
                end
                zetaBij = Bj(i,i) + 2*real(temp);
                temp = 0;
                for m = 1:1:N
                    if( m~=i )
                        for n = 1:1:N
                            if(n~=i)
                                temp = temp + conj(Vrf(m,j)) * Dj(m,n) * Vrf(n,j);
                            end
                        end
                    end
                end
                zetaDij = Dj(i,i)+2*real(temp);
                V_B = Bj * Vrf;
                V_D = Dj * Vrf;
                etaBij = V_B(i,j) - Bj(i,i)*Vrf(i,j);
                etaDij = V_D(i,j) - Dj(i,i)*Vrf(i,j);
                % 计算theta_1，theta_2
                cij = (1 + zetaDij) * etaBij - zetaBij * etaDij;
                zij = imag(2 * etaBij * etaDij);
                phij = 0;
                tt = asin(imag(cij)/abs(cij));
                if(real(cij)>=0) 
                    phij = tt; 
                else 
                    phij = pi - tt; 
                end
                theta_1 = -phij + asin(zij/abs(cij));
                theta_2 = pi - phij - asin(zij/abs(cij));
                % 判断最优theta
                V_RF1 = exp(-1j * theta_1);
                V_RF2 = exp(-1j * theta_2);
                f1 = N * trace(Aj^(-1)) -  N * (zetaBij + 2 * real(conj(V_RF1)*etaBij))/ (1+zetaDij+2*real(conj(V_RF1)*etaDij));
                f2 = N * trace(Aj^(-1)) - N * (zetaBij + 2 * real(conj(V_RF2)*etaBij))/ (1+zetaDij+2*real(conj(V_RF2)*etaDij));
                if(f1 < f2) 
                    theta_opt = theta_1; 
                else 
                    theta_opt = theta_2;
                end
                Vrf(i,j) = exp(-1j*theta_opt);
            end
        end
        fVrf_new = N * trace(pinv(Htilde * (Vrf * Vrf') * Htilde'));
        diff = abs((fVrf_new - fVrf_old)/fVrf_new);
    end
end















