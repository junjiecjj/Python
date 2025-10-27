function [dCap]= mqam_detector(M,r)
%Function to detect MQAM modulated symbols
%[dCap]= mqam_detector(M,r) detects the received MQAM signal points
%points - 'r'. M is the modulation level of MQAM
if(((M~=1) && ~mod(floor(log2(M)),2))==0), %M not a even power of 2
    error('Only Square MQAM supported. M must be even power of 2');
end
ref=constructQAM(M); %reference constellation for MQAM
[~,dCap]= iqOptDetector(r,ref); %IQ detection
end
