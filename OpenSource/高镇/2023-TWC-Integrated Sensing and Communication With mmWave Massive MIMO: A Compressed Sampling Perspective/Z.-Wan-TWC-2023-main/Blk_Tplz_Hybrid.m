function Phi = Blk_Tplz_Hybrid(N_ant, Q, L, N_RF, T_RF, T_GI, varargin)
    P_eff = Q - L + 1;
    if(T_RF > T_GI)
        N = ceil(P_eff/T_RF);
        F_set = exp(1i*2*pi*rand(N_RF,N_ant,N))/sqrt(N_ant);

        for i = 1:length(varargin)
            if(strcmpi(varargin{i}, 'pesudo'))
                F_set = varargin{i+1};
    %             S_set = varargin{i+2};
            end
        end

        S_set = zeros(P_eff,N_RF);
        for i = 1:N-1
            S_set((i-1)*T_RF+1:(i*T_RF-T_GI),:) = 2*(rand(T_RF-T_GI,N_RF)>0.5)-1;
                                                % exp(1i*2*pi*rand(length((i-1)*T+1:min(i*T-T_GI,P)),N_RF));
        end
        S_set((N-1)*T_RF+1:P_eff,:) = 2*(rand(P_eff-(N-1)*T_RF,N_RF)>0.5)-1;

        Phi = zeros(Q,L*N_ant);
        for i = 1:Q
            for j = 1:L
                idx = i-j+1;
                flag = (idx <= P_eff) && (idx >= 1) && ( (idx>(N-1)*T_RF) || (mod(idx-1,T_RF)<(T_RF-T_GI)));
                if(flag)
                    temp = S_set(idx,:)*F_set(:,:,ceil(idx/T_RF));
                    Phi(i,(j-1)*N_ant+1:j*N_ant) = temp/norm(temp,2);
                end
            end
        end
    else
        Phi = zeros(Q,L*N_ant);
        for i = 1:Q
            for j = 1:L
                idx = i-j+1;
                if((idx <= P_eff) && (idx >= 1))
                    temp = (2*(rand(1,N_RF)>0.5)-1)*exp(1i*2*pi*rand(N_RF,N_ant))/sqrt(N_ant);
                    Phi(i,(j-1)*N_ant+1:j*N_ant) = temp/norm(temp,2);
                end
            end
        end
    end
end