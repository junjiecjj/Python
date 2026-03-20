function sigma = Jfunc_inv(I)
%referencing Stephan ten Brink, "Design of low-density parity-check codes for modulation and detection," IEEE Transactions on Communications, vol. 52, no. 4, pp. 670 - 678, April 2004.
%The Jfunc in the Appendix

	Istar    =  0.3646;

	a_sigma1 =  1.09542;
	b_sigma1 =  0.214217;
	c_sigma1 =  2.33727;

	a_sigma2 =  0.706692;
	b_sigma2 =  0.386013;
	c_sigma2 = -1.75017;

	%assert(I>=0.0 && I<=1.0);

	sigma = zeros(size(I));
	
	idx = (I>=0.0 & I<=Istar);
	sigma(idx) = a_sigma1 .* I(idx) .* I(idx) + b_sigma1 .* I(idx) + c_sigma1 .* sqrt(I(idx));
	
	idx = (I>Istar & I<1.0);
	sigma(idx) = -a_sigma2 .* log( b_sigma2 .* (1.0-I(idx)) ) - c_sigma2 .* I(idx);
	
	%idx = (I==1.0); % I may >1
	%sigma(idx) = 100.0;
	
	idx = (I >=1.0);
	sigma(idx) = 20.0;
	
	sigma(sigma<0) = 0.0;
	%else
	%	'The input of Jfunc_inv is not between 0.0 and 1.0!'
	%end
end