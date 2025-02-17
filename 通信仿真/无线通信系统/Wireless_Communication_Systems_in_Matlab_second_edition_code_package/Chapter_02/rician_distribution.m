clearvars; clc;
N=100000; %Number of Samples
K_dB=[-80,0,3,6,10] ;%Rician K-factor in dB
K = 10.^(K_dB/10); %K factor in linear scale
plotStyle={'b-','r-','k-','g-','m-'};

for j = 1:length(K),
  K = 10.^(K_dB(j)/10);
  g1 = sqrt(K/(K+1)); %g1 for los path
  g2 =  1/sqrt(K+1); %g2 for specular components
  x = (g2*randn(1,N)+g1)+1i*(g2*randn(1,N)+g1);
  [Q,X] = hist(abs(x),1000);%histogram
  plot(X,Q/trapz(X,Q));%plot  estimated PDF
  display(['Average power : ',num2str(mean(abs(x).^2))]) ;
    
  %Trapz function gives the total area under the PDF curve. It is used as the normalization factor
  hold on;
  %Theoretical PDFs
  sigma0 = 1; rho = sqrt(K*2*sigma0^2);
  x=0:0.01:12;
  f= (x/sigma0^2).*exp(-(x.^2/(2*sigma0^2)+K^2)).*besseli(0,x*sqrt(2*K)/sigma0);
  plot(x,f,plotStyle{j});
  
  %Compute theoretical pdf and plot them
  r = 0:0.01:12; %rician rv
  Omega=1; %Total average power set to unity
  z = 2*r*sqrt(K*(K+1)/Omega); %for generating modified Bessel function
  I0_z= besseli(0,z); %modified Bessel function of first kind and order zero
  pdf = (2*r*(K+1)/Omega).*exp(-K-(r.^2*(K+1)/Omega)).*I0_z;%theoretical pdf
  plot(r,pdf,plotStyle{j+1});
end 


xlabel('Parameter - r');ylabel('f_R(r)');axis([0 6 0 0.7]);
