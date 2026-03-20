%% MIMO multipath channel
% Authors: 
% Yuhao Chi (yhchi@xidian.edu.cn)
% Yao Ge (yao.ge@ntu.edu.sg)
% ChatGPT-5
function H_TD_m = Get_channel_sparse(M, N, Nr, Ns, rho, fs, fs_N, P, index_D, dop, beta)
    MN = M * N;
    A = zeros(Nr, Ns, P);
    for i = 1 : P
        A(:, :, i) = relatedh(Nr, Ns, rho);
    end
    I_all = {}; J_all = {}; V_all = {};
    blk_id = 0;
    Jbase = []; Ibase = [];
    for ns = 1 : Ns
        for nr = 1 : Nr
            g_m_0 = Get_g(MN, fs, fs_N, P, index_D, dop, beta, A(nr,ns,:));
            L = size(g_m_0, 2) - 1;  
            if isempty(Jbase)
                cols_base = [1, (MN-L+1:MN)]; 
                colmat_base = repmat(cols_base, MN, 1);
                shifts = (0:MN-1)';                                 % MN×1
                colmat = mod(colmat_base + shifts - 1, MN) + 1;     % MN×(L+1)
                Jbase = reshape(colmat, [], 1);                     % MN*(L+1)×1
                Ibase = reshape(repmat((1:MN).', 1, L+1), [], 1);   % MN*(L+1)×1
            end
            Vblk = [g_m_0(:,1), fliplr(g_m_0(:,2:end))];
            Vblk = Vblk(:);
            Iblk = Ibase + (nr-1)*MN;
            Jblk = Jbase + (ns-1)*MN;
            blk_id = blk_id + 1;
            I_all{blk_id, 1} = Iblk;
            J_all{blk_id, 1} = Jblk;
            V_all{blk_id, 1} = Vblk;
        end
    end
    I_all = vertcat(I_all{:});
    J_all = vertcat(J_all{:});
    V_all = vertcat(V_all{:});
    H_TD_m = sparse(I_all, J_all, V_all, Nr*MN, Ns*MN); % MIMO time-domain channel (sparse)      
end

%%
function A = relatedh(Nr, Ns, rho)
    C_R = eye(Nr, Nr);
    for i = 1 : Nr
        for j = 1 : Nr
            if abs(i - j) < 30
                C_R(i, j) = rho^(abs(i - j));
            end
        end
    end
    C_R = sqrtm(C_R);
    C_T = eye(Ns, Ns);
    for i = 1 : Ns
        for j = 1 : Ns
            if abs(i - j) < 30
                C_T(i, j) = rho^(abs(i - j));
            end
        end
    end
    C_T = sqrtm(C_T);
    G = sqrt(1/2) * (randn(Nr, Ns) + randn(Nr, Ns)*1i);
    A = C_R * G * C_T;
end

%%
function g = Get_g(MN, fs, fs_N, P, index_D, dop, beta, h)
    tau_fix = [0 0.3 0.67 0.95 1.5] .* 1e-6;        % delay of the TU channel
    n_tap = 3;                                      % number of tap                           
    tau = tau_fix(1:P) + n_tap * fs_N / fs;                  
    pdb = h(:).';                                        
    if index_D
        theta = pi * (2*rand(1, P) - 1);
        vv = dop .* cos(theta);              
    else
        vv = zeros(1, P);                 
    end
    NN = 0:1/fs : max(tau) + n_tap * fs_N / fs;              
    t  = (0:MN-1) / fs;                                  
    A = exp(1j*2*pi*(t(:)*vv)) .* pdb;                 
    RRC = myrrc(beta, NN(:) - tau, fs/fs_N, 1);      
    B = exp(-1j*2*pi*(NN(:)*vv)) .* RRC;              
    g = A * B.';
end

function b = myrrc(beta, t1, fs, sps)
    t = t1 .* fs;
    x = (2*beta.*t).^2;
    b = zeros(size(t1), 'like', t1);
    mask = abs(1 - x) > sqrt(eps);
    tmp = sinc(t) .* (cos(pi*beta.*t)) ./ (1 - x) / sps;
    b(mask) = tmp(mask);
    b(~mask) = beta * sin(pi/(2*beta)) / (2*sps);
end