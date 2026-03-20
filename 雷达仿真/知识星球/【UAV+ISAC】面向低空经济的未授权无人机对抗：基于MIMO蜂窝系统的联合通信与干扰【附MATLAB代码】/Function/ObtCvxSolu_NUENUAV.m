function F_tilde = ObtCvxSolu_NUENUAV(A_1, gamma_1, ...
                         A_2, gamma_2, ...                            
                         A_3, gamma_3, ...
                         N_t, N_ue, N_uav, flag)
% run('D:\Code\Matlab\CVX\cvx\cvx_setup.m');
if flag
cvx_begin sdp quiet
    variable F_tilde(N_ue*N_t,N_ue*N_t) hermitian
    minimize(real(trace(F_tilde)));
    subject to
        for i_ue = 1:N_ue
            real(trace(A_1(:,:,i_ue)*F_tilde)) == gamma_1; 
        end
        for i_uav = 1:N_uav
            real(trace(A_2(:,:,i_uav)*F_tilde)) == gamma_2;            
        end    
        for i = 1:N_ue*N_t-1
            real(trace(A_3(:,:,i)*F_tilde)) >= gamma_3; 
        end
    F_tilde == hermitian_semidefinite(N_ue*N_t);
cvx_end
else
cvx_begin sdp quiet
    variable F_tilde(N_ue*N_t,N_ue*N_t) hermitian
    minimize(real(trace(F_tilde)));
    subject to
        for i_ue = 1:N_ue
            real(trace(A_1(:,:,i_ue)*F_tilde)) == gamma_1; 
        end
        for i_uav = 1:N_uav
            real(trace(A_2(:,:,i_uav)*F_tilde)) == gamma_2;            
        end    
    F_tilde == hermitian_semidefinite(N_ue*N_t);
cvx_end    
end

end