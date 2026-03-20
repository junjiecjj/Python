function I = Jfunc(sigma)
%referencing Stephan ten Brink, "Design of low-density parity-check codes for modulation and detection," IEEE Transactions on Communications, vol. 52, no. 4, pp. 670 - 678, April 2004.
%The Jfunc in the Appendix
	sigma_star =  1.6363;

	aJ1 = -0.0421061;
	bJ1 =  0.209252;
	cJ1 = -0.00640081;

	aJ2 =  0.00181491;
	bJ2 = -0.142675;
	cJ2 = -0.0822054;
	dJ2 =  0.0549608;

	I = zeros(size(sigma));
	
	idx = (sigma >=0.0 & sigma <= sigma_star);
	I(idx) = aJ1 .* sigma(idx) .* sigma(idx) .* sigma(idx) + bJ1 .* sigma(idx) .* sigma(idx) + cJ1 .* sigma(idx);
	
	idx = (sigma > sigma_star & sigma < 10.0 );
	I(idx) = 1.0 - exp(aJ2 .* sigma(idx) .* sigma(idx) .* sigma(idx) + bJ2 .* sigma(idx) .* sigma(idx) + cJ2 .* sigma(idx) + dJ2);
	
	idx	= (sigma >= 10.0);
	I(idx) = 1.0;
    I(I<0.0) = 0.0;
	I(I>1.0) = 1.0;
	%else
	%	'The input of Jfunc is smaller than 0.0!' 
	%end
    % I may >1 ?
end