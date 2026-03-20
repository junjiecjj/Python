function [MSE_Turbo_HFine, BER_Turbo_HFine, BERcw_Turbo_HFine] = Turbo_evo(beta, MaxESEit, InnerIt, SNRdB, HFinePath)
%turbo system evolution
    [dv, b, dc, d] = LDPC_check_distribution(HFinePath);
    
	BER_Turbo_HFine = zeros(MaxESEit,length(SNRdB));
	BERcw_Turbo_HFine = zeros(MaxESEit,length(SNRdB));
    MSE_Turbo_HFine = zeros(MaxESEit,length(SNRdB));
    for iii = 1:length(SNRdB)    
        N0 = 10^(-SNRdB(iii)/10);
        v2_Turbo = 1;
        I_memory_Turbo = 0;
        for it = 1:MaxESEit        
			if(InnerIt >= 2)
				I_memory_Turbo = 0;
			end
            tau2_Turbo = 1 ./ Turbo_phi(v2_Turbo, 1/N0, beta);
            [v2_Turbo, ~, I_memory_Turbo, BER_Turbo_HFine(it,iii), BERcw_Turbo_HFine(it,iii)] = LDPC_SE_memory(tau2_Turbo, I_memory_Turbo, InnerIt, dv, b, dc, d, 1e-6);
            MSE_Turbo_HFine(it,iii) = v2_Turbo;
			if(MSE_Turbo_HFine(it,iii)<=1e-20)
				MSE_Turbo_HFine((it+1):end,iii) = MSE_Turbo_HFine(it,iii);
				BER_Turbo_HFine((it+1):end,iii) = BER_Turbo_HFine(it,iii);
				BERcw_Turbo_HFine((it+1):end,iii) = BERcw_Turbo_HFine(it,iii);
				break;
			end
        end
		if(MSE_Turbo_HFine(end,iii)<=1e-20)
			break;
		end
    end
end