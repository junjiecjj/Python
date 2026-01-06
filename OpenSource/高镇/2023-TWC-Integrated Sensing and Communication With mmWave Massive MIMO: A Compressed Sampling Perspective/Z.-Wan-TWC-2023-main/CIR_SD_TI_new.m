function H = CIR_SD_TI_new(SterMtx_Tx, SterMtx_Rx, idx_tap, diag_alpha, NcNp, tau, tau_pulse, Ts, varargin)
    % 空间-延时(SD)域CIR，按[H_{idx_tap(1)-1},H_{idx_tap(2)-1},...,H_{idx_tap(L_tap)-1}]的顺序排列
    % 若是时变信道，则为一个时隙下的信道CIR， 时序的索引由idx_slot决定
  
    N_T = size(SterMtx_Tx,1);
    N_R = size(SterMtx_Rx,1);
    L_tap = length(idx_tap);
    diag_TV  = eye(NcNp);
    
    for i = 1:length(varargin)
        if(strcmpi(varargin{i}, 'tv'))
            Dplr = varargin{i+1};
            idx_slot = varargin{i+2};
            diag_TV  = diag(exp(-1i*2*pi*Dplr*idx_slot*Ts));
            if(i+2==length(varargin)); break; else; i = i + 2; end
        end
    end
    
    H = zeros(N_R,L_tap*N_T);
    for i = 1:L_tap
       diag_tau = diag(PSF_RaiCos((idx_tap(i)-1)*Ts-tau_pulse-tau,0.8,Ts,tau_pulse));
       H(:,(i-1)*N_T+1:i*N_T) = SterMtx_Rx * (diag_tau * diag_TV * diag_alpha) * SterMtx_Tx';
    end
end