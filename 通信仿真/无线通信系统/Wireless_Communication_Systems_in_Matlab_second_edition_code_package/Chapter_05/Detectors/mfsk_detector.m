function [dCap]= mfsk_detector(M,r,COHERENCE)
%Function to detect MFSK modulated symbols
%[dCap]= mfsk_detector(M,r,COHERENCE) detects the received MFSK 
%signal points - 'r'. M is the modulation level of MFSK.
%'COHERENCE' = 'COHERENT' or 'NONCOHERENT' specifies the type 
%of MFSK detection. The output dCap is the detected symbols.
if strcmpi(COHERENCE,'coherent'),
  phi= zeros(1,M); %phase=0 for coherent detection
  ref = complex(diag(exp(1i*phi)));%reference constellation
  [~,dCap]= minEuclideanDistance(real(r),ref);%coherent detection
else %for noncoherent MFSK
  [~,dCap]= max(abs(r),[],2);%envelope detection 
  dCap = dCap.';
end; end