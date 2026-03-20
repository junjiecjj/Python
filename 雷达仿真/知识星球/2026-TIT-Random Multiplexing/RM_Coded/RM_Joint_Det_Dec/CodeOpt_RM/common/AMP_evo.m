function [MSE_AMP_HFine, BER_AMP_HFine, BERcw_AMP_HFine] = AMP_evo(beta, MaxESEit, InnerIt, SNRdB, HFinePath)
	%AMP system evolution
	[dv, b, dc, d] = LDPC_check_distribution(HFinePath);
	
	BERcw_AMP_HFine = zeros(MaxESEit,length(SNRdB));
	BER_AMP_HFine = zeros(MaxESEit,length(SNRdB));
	MSE_AMP_HFine = zeros(MaxESEit,length(SNRdB));
	for iii = 1:length(SNRdB)
		N0 = 10^(-SNRdB(iii)/10);
		v2_AMP = 1;
		I_memory_AMP = 0;
		for it = 1:MaxESEit
			if(InnerIt >= 2)
				I_memory_AMP = 0;
			end
			tau2_AMP = beta * v2_AMP + N0;
			[~, v2_AMP, I_memory_AMP, BERcw_AMP_HFine(it,iii), BER_AMP_HFine(it,iii)] = LDPC_SE_memory(tau2_AMP, I_memory_AMP, InnerIt, dv, b, dc, d, 1e-6);
			MSE_AMP_HFine(it,iii) = v2_AMP;
			if(MSE_AMP_HFine(it,iii)<=1e-20)
				MSE_AMP_HFine((it+1):end,iii) = MSE_AMP_HFine(it,iii);
                BER_AMP_HFine((it+1):end,iii) = BER_AMP_HFine(it,iii);
                BERcw_AMP_HFine((it+1):end,iii) = BERcw_AMP_HFine(it,iii);
				break;
			end
		end
		if(MSE_AMP_HFine(end,iii)<=1e-20)
			break;
		end
	end
end
