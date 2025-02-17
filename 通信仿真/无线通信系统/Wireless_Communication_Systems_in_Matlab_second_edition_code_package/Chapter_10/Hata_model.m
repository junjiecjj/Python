function PL = Hata_model(fc,d,hb,hm,envType)
%Hata model for propagation loss
%   fc - frequency in MHz (single value)
%   d - array of distances (in Km) to simulate
%   hb - effective base station height in meters
%   hm - effective mobile height in meters
%   envType - 'metro','smallcity','suburban' or 'open'
%   returns the path loss (PL) in dB
%Warning: Hata model is valid only if the model parameters
%falls within certain range of values(see text). This function 
%has not implemented those range checks.
envType=lower(envType);
switch envType
 case 'metro',
   C=0;
   if fc<=200, aHm=8.29*(log10(1.54*hm))^2-1.1;
   else aHm=3.2*(log10(11.75*hm))^2-4.97;
   end
 case 'smallcity', 
   C=0; aHm = (1.1*log10(fc)-0.7)*hm-(1.56*log10(fc)-0.8);
 case 'suburban',
   aHm = (1.1*log10(fc)-0.7)*hm-(1.56*log10(fc)-0.8);
   C=-2*(log10(fc/28)).^2-5.4;
 case 'open',
   aHm = (1.1*log10(fc)-0.7)*hm-(1.56*log10(fc)-0.8);
   C=-4.78*(log10(fc)).^2+18.33*log10(fc)-40.98;
 otherwise ,  error('Invalid model selection');
end
 A = 69.55 + 26.16*log10(fc) - 13.82*log10(hb)-aHm;
 B = 44.9 - 6.55*log10(hb);
 PL=A+B*log10(d)+C;
end