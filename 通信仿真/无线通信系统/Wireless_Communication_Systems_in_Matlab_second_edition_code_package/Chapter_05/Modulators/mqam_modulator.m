function [s,ref]=mqam_modulator(M,d)
%Function to MQAM modulate the vector of data symbols - d
%[s,ref]=mqam_modulator(M,d) modulates the symbols defined by the vector d
% using MQAM modulation, where M specifies order of M-QAM modulation and
% vector d contains symbols whose values range 1:M. The output s is modulated
% output and ref represents reference constellation that can be used in demod
if(((M~=1) && ~mod(floor(log2(M)),2))==0), %M not a even power of 2
    error('Only Square MQAM supported. M must be even power of 2');
end
  ref=constructQAM(M); %construct reference constellation
  s=ref(d); %map information symbols to modulated symbols
end