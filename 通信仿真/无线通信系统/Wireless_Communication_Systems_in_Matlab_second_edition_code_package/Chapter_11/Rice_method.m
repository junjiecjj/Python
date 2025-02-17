function Rice_method(T_sim,T_s,DOPPLER_PSD,N_i,f_max,f_c,sigma_0,...
   rho,f_rho,theta_rho,PLOT)
%T_sim = Simulation time, T_s = Sampling interval
%DOPPLER_PSD: Defines type of PSD used for Doppler spreading
%   'JAKES' - Jakes Power Spectral Density
%   'GAUSS' - Gaussian Power Spectral Density
%N_i = Number of harmonic functions (Ni=N1=N2 for simplicity)
%f_max = Maximum Doppler frequency, f_c = carrier frequency
%sigma_0 = average power of the real deterministic Gaussian process mu_i(t)
%----LOS component parameters---
%rho = amplitude of LOS component m(t),
%f_rho = Doppler frequency of LOS component m(t)
%theta_rho = phase of LOS component m(t)
%PLOT = 1 plots different graphs
%Example: Rice_method(2,0.0001,'JAKES',200,100,1000,1,1,5,0.5,1)

%Generate c_i, f_i and theta_i_n using equations for MEDS method
%independently for both inphase and quadrature arm
[c_1,f_1,theta_1] = param_MEDS(DOPPLER_PSD,N_i,f_max,f_c,sigma_0);
[c_2,f_2,theta_2] = param_MEDS(DOPPLER_PSD,N_i+1,f_max,f_c,sigma_0);

no_samples = ceil(T_sim/T_s); %#samples generated for simulation
t=(0:no_samples-1)*T_s; %discrete-time instants

mu_i = 0; mu_q = 0;
for n=1:N_i,%computation for the two branches
    mu_i = mu_i + c_1(n)*cos(2*pi*f_1(n)*t+theta_1(n));
    mu_q = mu_q + c_2(n)*cos(2*pi*f_2(n)*t+theta_2(n));    
end
%Generate LOS components and add it to the gaussian processes
m_i = rho*cos(2*pi*f_rho*t+ theta_rho);
m_q = rho*sin(2*pi*f_rho*t+ theta_rho);
xi_t = abs(mu_i+m_i+1i*(mu_q+m_q));

if PLOT,
  subplot(1,2,1); plot(t,20*log10(xi_t));
  xlabel('time (s)'), ylabel('amplitude');title('Fading samples');
  %use the function given in section 1.3.4 to plot the spectrum
  [S_mu_f,F] = freqDomainView(mu_i,1/T_s,'double');
  subplot(1,2,2); plot(F,abs(S_mu_f)); axis tight;
  xlim([-f_max-50,f_max+50]);title('PSD of \mu_i(t)');
  xlabel('frequency'); ylabel('S_{\mu_i\mu_i(f)}');
end