function [dCap]= mpam_detector(M,r)
%Function to detect MPAM modulated symbols
%[dCap]= mqam_detector(M,r) detects the received MPAM signal points
%points - 'r'. M is the modulation level of MPAM
m=1:1:M; Am=complex(2*m-1-M); %all possibe amplitude levels
ref = Am; %reference constellation for MPAM
[~,dCap]= iqOptDetector(r,ref); %IQ detection
end
