function [snr, rate] = MIMOShannonlimit(beta, target_rate, type)
%target_rate: the Shannon limit for the target rate
%type: "Gaussian" for AMP Gaussian
%      "AMP-QPSK" for AMP QPSK
%      "Turbo-QPSK" for Turbo QPSK
max_snr = 1e+6;
min_snr = 1e-6;
switch(type)
	case {'Gaussian'},
		while(abs(max_snr-min_snr)>1e-6)
			snr = 0.5*(max_snr+min_snr);
			rate    = Gaussian_MIMO_Capacity(beta, 1/snr);
			if(rate > target_rate)
				max_snr = snr;
			else
				min_snr = snr;
			end
		end
		snr  = 0.5*(max_snr+min_snr);
		rate = Gaussian_MIMO_Capacity(beta, 1/snr);
	
	case{'AMP-QPSK'},
			rate = Rate_AMP_QPSK(beta, max_snr);
		if(target_rate>rate)
			error('Target rate is too large for AMP-QPSK!');
		end
		while(abs(max_snr-min_snr)>1e-6)
			snr  = 0.5*(max_snr+min_snr);
			rate = Rate_AMP_QPSK(beta, snr);
			if(rate > target_rate)
				max_snr = snr;
			else
				min_snr = snr;
			end
		end
		snr  = 0.5*(max_snr+min_snr);
		rate = Rate_AMP_QPSK(beta, snr);
	
	case{'Turbo-QPSK'},
		rate = Rate_Turbo_QPSK(beta, max_snr);
		if(target_rate > rate)
			error('Target rate is too large for Turbo-QPSK!');
		end
		while(abs(max_snr-min_snr)>1e-6)
			snr  = 0.5*(max_snr+min_snr);
			rate = Rate_Turbo_QPSK(beta, snr);
			if(rate > target_rate)
				max_snr = snr;
			else
				min_snr = snr;
			end
		end
	
	otherwise,
		error('Input type invalid!');
end