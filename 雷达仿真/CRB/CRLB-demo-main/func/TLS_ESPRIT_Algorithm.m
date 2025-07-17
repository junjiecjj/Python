% function Est_para = TLS_ESPRIT_Algorithm(Y_ll_bar, L, nargin1, nargin2, nargin3)
function psi = TLS_ESPRIT_Algorithm(Y_ll_bar, L)
% TLS-ESPRIT Algorithm

    K_sub = size(Y_ll_bar,1);
    
    Y_sub1 = Y_ll_bar(1:end-1,:);
    Y_sub2 = Y_ll_bar(2:end,:);
    Z_mtx = [Y_sub1; Y_sub2];
    R_ZZ = Z_mtx*Z_mtx';
    
%     [Us, ~, ~] = svd(R_ZZ);
%     Es = Us(:,L);
    [Es, ~, ~] = svds(R_ZZ, L);
    
    Exy = [Es(1:K_sub-1,:), Es(K_sub:end,:)];
    Exy_Conj = [Es(1:K_sub-1,:), Es(K_sub:end,:)]';
    EE = Exy_Conj*Exy;
    
    [~, ~, E] = svd(EE);	% svd
    E12 = E(1:L, L+1:end);
    E22 = E(L+1:end, L+1:end);
    Psi = -E12/E22;
    psi = eig(Psi);
%     switch nargin
%         case 5
%             Est_para = angle(psi)/(-2*pi*nargin1*nargin2/nargin3);	% Estimate delays
%         case 3
%             Est_para = angle(psi)/(2*pi*nargin1);                   % Estimate Doppler shifts
%         otherwise
%             error('Error: invalid INPUT!');
%     end
end