function [dCap]= mpsk_detector(M,r)
%Function to detect MPSK modulated symbols
%[dCap]= mpsk_detector(M,r) detects the received MPSK signal points
%points - 'r'. M is the modulation level of MPSK
ref_i= 1/sqrt(2)*cos(((1:1:M)-1)/M*2*pi); 
ref_q= 1/sqrt(2)*sin(((1:1:M)-1)/M*2*pi);
ref = ref_i+1i*ref_q; %reference constellation for MPSK
[~,dCap]= iqOptDetector(r,ref); %IQ detection
end
