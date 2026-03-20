%% Generalized Schur Algorithm
%% Takes (n-1)'th degree polynomials alpha_0 and beta_0, such that
%% phi_0=alpha_0/beta_0 is a Schur function.
%% Returns the n'th Schur polynomials (of degree n-1) xi_n, eta_n, obtained by
%% running n steps of the Schur algorithm on phi_0.
%% Based on [1] G. S. Ammar and W. B. Gragg, "The Generalized Schur Algorithm
%% for the Superfast Solution of Toeplitz Systems"
% Copyright (c) 2018: Thomas Lundgaard Hansen.
% This software is being released under the MIT license (see LICENSE file). 
function [xi_n, eta_n, gamma] = genschur(alpha_0_in, beta_0_in) %#codegen
	n = length(alpha_0_in);

	% Check inputs
	alpha_0 = alpha_0_in(:).';
	beta_0 = beta_0_in(:).';

	% For small n use the recurrence relations in the beginning of [1, Sec 3.1]
	if n <= 32
		alpha_0 = complex(real(alpha_0), imag(alpha_0));
		beta_0 = complex(real(beta_0), imag(beta_0));
		[xi_n, eta_n, gamma] = genschur_recurrence(alpha_0, beta_0);
		return;
	end

	%% Trivial special case n=1
	%if n == 1
		%xi_n = alpha_0 / beta_0;
		%eta_n = 1;
		%gamma = xi_n;
		%return;
	%end

	% Splitting n = n1+n2
	n1 = floor(n/2);
	n2 = n - n1;

	% Step A:
	% Obtain n1'th Schur polynomials of phi_0
	% I.e. find T_0,n1 in [1]
	[xi_n1, eta_n1, gamma_lower] = genschur(alpha_0(1:n1), beta_0(1:n1));

	% Step B (step 1 in [1]):
	% Obtain first n1 coefficients of n1'th tail of phi_0
	% I.e. find phi_n1 in [1]
	% Products of polynomials are convolutions of coefficient vectors
	N = 2^ceil(log2(n + n1 - 1));
	%N = n + n1 - 1;
	N0 = N - length(eta_n1);
	shiftvec = exp( (1j*2*pi*(N0+1)/N) * (0:N-1));
	FFTalpha_0 = fft(alpha_0,N);
	FFTbeta_0 = fft(beta_0,N);
	FFTeta_n1 = fft(eta_n1,N);
	FFTxi_n1 =  fft(xi_n1,N);
	%alpha_n1 = conv(alpha_0, eta_n1) - conv(beta_0, xi_n1);
	alpha_n1 = ifft(FFTalpha_0 .* FFTeta_n1 - FFTbeta_0 .* FFTxi_n1);
	alpha_n1 = alpha_n1(n1+1:n1+n2);	% Divide by lambda^(n1), retain n2 coefs
	%beta_n1 = conv(beta_0, conj(fliplr(eta_n1))) ...
				%- conv(alpha_0, conj(fliplr(xi_n1)));
	beta_n1 = ifft( (FFTbeta_0 .* conj(FFTeta_n1) ...
				- FFTalpha_0 .* conj(FFTxi_n1)).*shiftvec );
	beta_n1 = beta_n1(n1:n1+n2-1);		% Divide by lambda^(n1-1), retain n2 coefs

	% Step C (doubling step, step 2 in [1]):
	% Obtain n/2'th Schur polynomials of phi_n/2
	% I.e. find T_n/2,n/2 in [1]
	[xi_n1n2, eta_n1n2, gamma_upper] = genschur(alpha_n1, beta_n1);
	FFTxi_n1n2 = fft(xi_n1n2, N);
	FFTeta_n1n2 = fft(eta_n1n2, N);

	% Step D (step 3 in [1]):
	% Obtain n'th Schur polynomials of phi_0 by composition of T_0,n/2 and
	% T_n/2,n/2.
	% I.e. find T_0,n in [1]
	%xi_n = [0, conv(conj(fliplr(eta_n1)), xi_n1n2)] ...
				%+ [conv(xi_n1, eta_n1n2), 0];
	xi_n = [0, ifft(conj(FFTeta_n1) .* shiftvec .* FFTxi_n1n2)];
	xi_n = xi_n(1:n);
	tmp = ifft( FFTxi_n1 .* FFTeta_n1n2 );
	xi_n(1:n-1) = xi_n(1:n-1) + tmp(1:n-1);
	%eta_n = [0, conv(conj(fliplr(xi_n1)), xi_n1n2)] ...
				%+ [conv(eta_n1, eta_n1n2), 0];
	eta_n = [0, ifft(conj(FFTxi_n1) .* shiftvec .* FFTxi_n1n2)];
	eta_n = eta_n(1:n);
	tmp = ifft( FFTeta_n1 .* FFTeta_n1n2 );
	eta_n(1:n-1) = eta_n(1:n-1) + tmp(1:n-1);

	% Compose gamma
	gamma = [gamma_lower; gamma_upper];
end

%% Inputs alpha_0 and beta_0 are zero-padded FFT'ed of length 2n.
%% Outputs xi_n and eta_n are zero-padded FFT'ed of length 2n.
%function [xi_n, eta_n, gamma] = gen_schur_alg(alpha_0, beta_0)
	%assert( rem(length(alpha_0), 2) == 0 );
	%assert( length(alpha_0)==length(beta_0) );
	%n = length(alpha_0)/2;
	%assert( n==1 || rem(n,2)==0 );
%
	%% Trivial special case n=1
	%if n == 1
		%xi_n = ifft(alpha_0) ./ ifft(beta_0);
		%xi_n = xi_n(1);
		%eta_n = 1;
		%gamma = xi_n;
		%xi_n = fft(xi_n, 2*n);
		%eta_n = fft(eta_n, 2*n);
		%return;
	%end
%
	%% Step A:
	%% Obtain n/2'th Schur polynomials of phi_0
	%% I.e. find T_0,n/2 in [1]
	%%alpha_in = (alpha_0(1:2:end) + alpha_0(2:2:end)) / 2;
	%%beta_in = (beta_0(1:2:end) + beta_0(2:2:end)) / 2;
	%tmp = ifft(alpha_0);
	%alpha_in = fft(tmp(1:n/2), n);
	%tmp = ifft(beta_0);
	%beta_in = fft(tmp(1:n/2), n);
	%[xi_n2, eta_n2, gamma_lower] = gen_schur_alg(alpha_in, beta_in);
%
	%% Step B (step 1 in [1]):
	%% Obtain first n/2 coefficients of n/2'th tail of phi_0
	%% I.e. find phi_n/2 in [1]
	%% Products of polynomials are convolutions of coefficient vectors
	%%alpha_n2 = conv(alpha_0, eta_n2) - conv(beta_0, xi_n2);
	%%alpha_n2 = alpha_n2(n/2+1:n);	% Divide by lambda^(n/2), retain n/2 coefs
%
	%alpha_n2 = ifft(alpha_0.*fft(ifft(eta_n2),2*n) - beta_0.*fft(ifft(xi_n2),2*n));
	%alpha_n2 = alpha_n2(n/2+1:n);	% Divide by lambda^(n/2), retain n/2 coefs
	%alpha_n2 = fft(alpha_n2, n);
%
	%tmp1 = ifft(eta_n2);
	%tmp2 = ifft(xi_n2);
	%%beta_n2 = conv(ifft(beta_0), [0,conj(fliplr(tmp1(1:n/2)))]) ...
				%%- conv(ifft(alpha_0), [0,conj(fliplr(tmp2(1:n/2)))]);
	%%beta_n2 = beta_n2(n/2+1:n);		% Divide by lambda^(n/2), retain n/2 coefs
	%%beta_n2 = fft(beta_n2, n);


	%shiftvec = exp(-j*2*pi*[0:n-1]*n/2/n);
	%%out = conv(ifft(beta_0), conj(fliplr(tmp1(1:n/2))));
	%out = ifft(beta_0(1:2:end) .* fft(conj(fliplr(tmp1(1:n/2))), n));
	%%out = ifft( beta_0(1:2:end) .* fliplr(eta_n2.*shiftvec) );
	%out = out(n/2:n-1);				% Divide by lambda^(n/2), retain n/2 coefs
	%out = fft(out, n);
	%beta_n2 = - conv(ifft(alpha_0), conj(fliplr(tmp2(1:n/2))));
	%beta_n2 = beta_n2(n/2:n-1);		% Divide by lambda^(n/2), retain n/2 coefs
	%beta_n2 = out + fft(beta_n2, n);
%
	%% Step C (doubling step, step 2 in [1]):
	%% Obtain n/2'th Schur polynomials of phi_n/2
	%% I.e. find T_n/2,n/2 in [1]
	%[xi_n2n2, eta_n2n2, gamma_upper] = gen_schur_alg(alpha_n2, beta_n2);
%
	%% Step D (step 3 in [1]):
	%% Obtain n'th Schur polynomials of phi_0 by composition of T_0,n/2 and
	%% T_n/2,n/2.
	%% I.e. find T_0,n in [1]
	%tmp1 = ifft(eta_n2);
	%tmp2 = ifft(xi_n2);
	%xi_n = conv([0,conj(fliplr(tmp1(1:n/2)))], ifft(xi_n2n2)) ...
				%+ [conv(tmp2(1:n/2), ifft(eta_n2n2)), 0];
	%eta_n = conv([0,conj(fliplr(tmp2(1:n/2)))], ifft(xi_n2n2)) ...
				%+ [conv(tmp1(1:n/2), ifft(eta_n2n2)), 0];
	%xi_n = fft(xi_n, 2*n);		% Zero-pad to length 2n
	%eta_n = fft(eta_n, 2*n);	% Zero-pad to length 2n
	%gamma = [gamma_lower; gamma_upper];
%end


