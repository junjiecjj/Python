%Simulate receieved signal samples due to Ricean flat-fading
Nsym = 10^5; %number of received signal samples to generate
K_factors = [0 3 7 12 20]; %Ricean K factors
colors={'b','r','k','g','m'}; index=1;

for K=K_factors, %simulate for each K factors    
  g1 = sqrt(K/(2*(K+1))); g2 =  sqrt(1/(2*(K+1)));
  %Generate 10^5 Rice fading samples with unity average power
  r = (g2*randn(1,N)+g1)+1i*(g2*randn(1,N)+g1); 
  display(['Average power : ',num2str(mean(abs(r).^2))]) ;
  
  [val,bin] = hist(abs(r),50); %PDF of envelope the samples
  plot(bin,val/trapz(bin,val),[colors{index} 'O']);%normalized pdf  
  %Compute theoretical pdf and plot them
  x = 0:0.05:3; %Ricean rv
  Omega=1; %Total average power set to unity
  z = 2*x*sqrt(K*(K+1)/Omega);%to generate modified Bessel function
  I0_z= besseli(0,z);%modified Bessel function of first kind
  pdf = (2*x*(K+1)/Omega).*exp(-K-(x.^2*(K+1)/Omega)).*I0_z;  
  hold on; plot(x,pdf,colors{index});index=index+1;  
end
title('PDF of envelope of received signal');xlabel('x');ylabel('f_{\xi}(x)');