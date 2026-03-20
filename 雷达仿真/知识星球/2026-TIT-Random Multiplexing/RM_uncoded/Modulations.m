%% OFDM, OTFS, AFDM or RM
% ------------------------------------------------------------------
% Input:
% s: signal s to be modulated/demodulated
% is_inv: 0 means modulation, 1 means its inverse
% info: struct with filed 'type'
% (1a) type 'OFDM_j': info includes filed 'N_x'
% (1b) type 'OFDM_p': info includes filed 'MN', 'N_s'
% (2) type 'OTFS': info includes filed 'M', 'N', 'N_s'
% (3a) type 'ADDM_j': info includes filed 'N_x', 'c_1', 'c_2'
% (3b) type 'AFDM_p': info includes filed 'MN', 'N_s', 'c_1', 'c_2'
% (4) type 'RM': info includes filed 'N_x', 'RM_type', 'index'
% (5) type 'None': No modulations
% ------------------------------------------------------------------
% Difference: {(1a) and (1b)} or {(3a) and (3b)}
% "_j" means joint modulation scheme:
% The data for all antennas is modulated collectively as a single block.
% "_p" means per-antenna modulation scheme:
% The data is treated as distinct blocks for each antenna, and each block 
% is modulated independently.
% For SISO systems, (1a)=(1b), (3a)=(3b)
% ------------------------------------------------------------------ 
function x = Modulations(s, info, is_inv)
    type = info.type;
    if strcmpi(type, 'OFDM_j')
        N_x = info.N_x;
        if is_inv
            x = fft(s) / sqrt(N_x);
        else
            x = ifft(s) * sqrt(N_x);
        end
    elseif strcmpi(type, 'OFDM_p')
        MN = info.MN;
        N_s = info.N_s;
        S = reshape(s, MN, N_s);
        if is_inv
            X = fft(S, MN, 1) / sqrt(MN);
        else
            X = ifft(S, MN, 1) * sqrt(MN);
        end
        x = X(:);
    elseif strcmpi(type, 'OTFS')
        M = info.M;
        N = info.N;
        N_s = info.N_s;
        S = reshape(s, M, N, N_s);
        if is_inv
            X = fft(S, [], 2) / sqrt(N);
        else
            X = ifft(S, [], 2) * sqrt(N);
        end
        x = X(:);
    elseif strcmpi(type, 'AFDM_j')
        N_x = info.N_x;
        c1 = info.c1;
        c2 = info.c2;
        aa = (1:N_x).';
        Phi_c1 = exp(-1j*2*pi*c1*(aa-1).^2);
        Phi_c2 = exp(-1j*2*pi*c2*(aa-1).^2);
        if is_inv
            x1 = Phi_c1 .* s;
            x2 = fft(x1) / sqrt(N_x);
            x = Phi_c2 .* x2;
        else
            x1 = conj(Phi_c2) .* s;
            x2 = ifft(x1) * sqrt(N_x);
            x = conj(Phi_c1) .* x2;
        end
    elseif strcmpi(type, 'AFDM_p')
        MN = info.MN;
        N_s = info.N_s;
        c1 = info.c1;
        c2 = info.c2;
        aa = (1:MN).';
        Phi_c1 = exp(-1j*2*pi*c1*(aa-1).^2);
        Phi_c2 = exp(-1j*2*pi*c2*(aa-1).^2);
        if is_inv
            S = reshape(s, MN, N_s);
            X1 = Phi_c1 .* S;
            X2 = fft(X1, MN, 1) / sqrt(MN);
            X = Phi_c2 .* X2;
            x = X(:);
        else
            S = reshape(s, MN, N_s);
            X1 = conj(Phi_c2) .* S;
            X2 = ifft(X1, MN, 1) * sqrt(MN);
            X = conj(Phi_c1) .* X2;
            x = X(:);
        end
    elseif strcmpi(type, 'RM')
        % see "Random_transform.m" for details
        x = Random_transform(s, info.rm_type, is_inv, info.index);
    elseif strcmpi(type, 'None')
        x = s;
    else 
        error('Modulation Error: "%s" is not supported currently!', type);
    end
end

% ------------------------------------------------------------------
% % Modulation matrices:
% % OFDM (joint)
% F = dftmtx(N_x) / sqrt(N_x);
% U_ofdm = F';                            % OFDM modulation matrix
% % OFDM (per-antenna)
% F = dftmtx(MN) / sqrt(MN);
% U_ofdm = kron(eye(N_s), F');            % OFDM modulation matrix
% % OTFS
% A1 = kron(dftmtx(N)/sqrt(N), eye(M));          
% U_otfs = kron(eye(N_s), A1');           % OTFS modulation matrix
% % AFDM (joint)
% Epsilon = N;                            % 0 (>0): integer (fractional) Doppler shift
% Doppler_taps_max = round(dop*N/delta_f);
% c1 = (2*(Doppler_taps_max+Epsilon)+1) / (2*N_x);
% c2 = 1e-5;                              % c2 should be much smaller than 1/(2*N_x)
% aa = (1:N_x).';
% Phi_c1 = exp(-1j*2*pi*c1*(aa-1).^2);
% Phi_c2 = exp(-1j*2*pi*c2*(aa-1).^2);
% F = dftmtx(N_x) / sqrt(N_x);
% U_afdm = (Phi_c2 .* F .* Phi_c1.')';    % AFDM modulation matrix
% % AFDM (per-antenna)         
% Epsilon = N;                            
% Doppler_taps_max = round(dop*N/delta_f);
% c1 = (2*(Doppler_taps_max+Epsilon)+1) / (2*MN);
% c2 = 1e-5;                              % c2 should be much smaller than 1/(2*MN)                              
% aa = (1:MN).';
% Phi_c1 = exp(-1j*2*pi*c1*(aa-1).^2);
% Phi_c2 = exp(-1j*2*pi*c2*(aa-1).^2);
% F = dftmtx(MN) / sqrt(MN);
% A2 = Phi_c2 .* F .* Phi_c1.';
% U_afdm = kron(eye(N_s), A2');           % AFDM modulation matrix