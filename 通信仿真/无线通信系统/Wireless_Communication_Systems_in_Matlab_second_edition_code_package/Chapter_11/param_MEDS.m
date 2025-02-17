function [c_i_n,f_i_n,theta_i_n] = param_MEDS(DOPPLER_PSD,N_i,f_max,f_c,sigma_0)
%Generate parameters for Sum of Sinusoid (SOS) simulator using the Method of
%exact Doppler spread (MEDS)
%DOPPLER_PSD: Defines type of PSD used for Doppler spreading
%   'JAKES' - Jakes Power Spectral Density
%   'GAUSS' - Gaussian Power Spectral Density
%N_i = Number of harmonic functions
%f_max = Maximum Doppler frequency
%f_c = carrier frequency
%sigma_0 = average power of the real deterministic Gaussian process mu_i(t)

%Generate Doppler coefficients c_i and Doppler frequencies f_i using MEDS method
n=(1:N_i).';
if strcmpi(DOPPLER_PSD,'JAKES'),
   c_i_n = sigma_0*sqrt(2/N_i)*ones(size(n));
   f_i_n = f_max*sin(pi/(2*N_i)*(n-1/2));   
elseif strcmpi(DOPPLER_PSD,'GAUSS'),
   c_i_n = sigma_0*sqrt(2/N_i)*ones(size(n));
   f_i_n=f_c/sqrt(log(2))*erfinv((2*n-1)/(2*N_i));   
else error('Invalid DOPPLER_PSD specified');
end
theta_i_n=2*pi*rand(N_i,1); %randomized Doppler phases